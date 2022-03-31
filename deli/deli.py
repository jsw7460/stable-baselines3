import io
import pathlib
from collections import deque
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch as th

from buffers import TrajectoryBuffer
from policies import DeliPolicy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import (
    check_for_correct_spaces,
)

DEQUE = partial(deque, maxlen=100)

class Deli(OffPolicyAlgorithm):
    def __init__(
        self,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 0,       # Do not change for Deli
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,

        gumbel_ensemble: bool = False,
        gumbel_temperature: float = 0.5,
        expert_data_path: str = None,       # If evaluation, None
        dropout: float = 0.1,
        additional_dim: int = 1,
        vae_feature_dim: int = 256,
        latent_dim: int = 128,
    ):

        super(Deli, self).__init__(
            "MlpPolicy",
            env,
            DeliPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box),
            without_exploration=True,
            gumbel_ensemble=gumbel_ensemble,
            gumbel_temperature=gumbel_temperature,
            dropout=dropout,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entroclearpy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer = None

        self.additional_dim = additional_dim
        self.vae_feature_dim = vae_feature_dim
        self.latent_dim = latent_dim

        self.ent_coef_losses, self.ent_coefs = DEQUE(), DEQUE()
        self.log_likelihood = DEQUE()

        if expert_data_path is not None:
            self.replay_buffer = TrajectoryBuffer(
                expert_data_path=expert_data_path,
                observation_space=env.observation_space,
                action_space=env.action_space,
                max_traj_len=1000,
                device=self.device
            )

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self._create_aliases()
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)

    def _create_aliases(self) -> None:
        self.policy_kwargs["dropout"] = self.dropout
        self.policy_kwargs["latent_dim"] = self.latent_dim
        self.policy = DeliPolicy(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)
        self._convert_train_freq()

        self.actor = self.policy.actor
        self.vae = self.policy.vae.to(self.policy.device)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            len_subtraj = np.random.randint(low=5, high=100)
            replay_data = self.replay_buffer.subtraj_sample(batch_size, len_subtraj)

            # Define the input data by concatenating the ingradients.
            history_tensor = th.cat((replay_data.history.observations, replay_data.history.actions), dim=2)
            future_tensor = th.cat((replay_data.future.observations, replay_data.future.actions), dim=2)

            history_latent, future_latent, history_stat, future_stat = self.vae(history_tensor, future_tensor)

            latent = th.cat((history_latent, future_latent), dim=1)
            assert th.isnan(latent).sum() == 0

            # NOTE ---- Start: Latent vector KL-loss
            history_mu, history_log_std = history_stat
            future_mu, future_log_std = future_stat
            history_std, future_std = th.exp(history_log_std), th.exp(future_log_std)

            history_kl_loss = th.log(1 / history_std) + (history_std ** 2 + history_mu ** 2) / 2 - 0.5
            future_kl_loss = th.log(1 / future_std) + (future_std ** 2 + future_mu ** 2) / 2 - 0.5

            kl_loss = 0.5 * (history_kl_loss + future_kl_loss).mean()
            # NOTE ---- End: Latent vector KL-loss

            # NOTE ---- Start: entropy coefficient loss
            # Action by the current actor for the sampled state
            policy_input = th.cat((replay_data.observations, latent.detach()), dim=1)
            actions_pi, log_prob = self.actor.action_log_prob(policy_input)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                self.ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            self.ent_coefs.append(ent_coef.item())
            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
            # NOTE ---- End: entropy coefficient loss

            # NOTE ---- Start: Deli
            policy_input = th.cat((replay_data.observations, latent), dim=1)
            action_log_prob = self.actor.get_log_prob(policy_input, replay_data.actions)
            self.log_likelihood.append(action_log_prob.mean().item())
            loss = -action_log_prob.mean()
            loss = loss + kl_loss

            self.actor.zero_grad()
            loss.backward()
            self.actor.optimizer.step()

            # NOTE ---- End: Deli

            # # Update target networks
            # if gradient_step % self.target_update_interval == 0:
            #     polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(self.ent_coefs))
        self.logger.record("train/likelihood", np.mean(self.log_likelihood))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "Deli",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(Deli, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _excluded_save_params(self) -> List[str]:
        return super(Deli, self)._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables

    @classmethod
    def load_deli(
            cls,
            path: Union[str, pathlib.Path, io.BufferedIOBase],
            env: Optional[GymEnv] = None,
            device: Union[th.device, str] = "auto",
            custom_objects: Optional[Dict[str, Any]] = None,
            print_system_info: bool = False,
            **kwargs,
    ) -> "BaseAlgorithm":
        """
        Load the model from a zip-file

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param kwargs: extra arguments to change the model when loading
        """
        # if print_system_info:
        #     print("== CURRENT SYSTEM INFO ==")
        #     get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path, device=device, custom_objects=custom_objects
        )

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        # noinspection PyArgumentList
        model = cls(  # pytype: disable=not-instantiable,wrong-keyword-args
            env=env,
            device=device,
            _init_setup_model=False,  # pytype: disable=not-instantiable,wrong-keyword-args
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        # put state_dicts back in place
        model.set_parameters(params, exact_match=True, device=device)

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, name + ".data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # pytype: disable=attribute-error
        return model


