import io
import pathlib
from collections import deque
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch as th

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import (
    check_for_correct_spaces,
)
from .policies import DeliG4Policy
from ..common.buffers import HindsightBuffer
from ..deli.features_extractor import HistoryVAE

th.autograd.set_detect_anomaly(True)

DEQUE = partial(deque, maxlen=100)


class DeliG4(OffPolicyAlgorithm):
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
        max_traj_len: int = -1,
        subtraj_len: int = 10,
        learn_latent: bool = False,         # If true, the gradient for latent vector flows by the actor.
    ):
        super(DeliG4, self).__init__(
            "DeliG4Policy",
            env,
            DeliG4Policy,
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

        self.subtraj_len = subtraj_len
        self.additional_dim = additional_dim
        self.vae_feature_dim = vae_feature_dim
        self.latent_dim = latent_dim
        self.learn_latent = learn_latent

        self.ent_coef_losses, self.ent_coefs = DEQUE(), DEQUE()
        self.log_likelihood = DEQUE()
        self.history_mues, self.history_stds = DEQUE(), DEQUE()
        self.goal_mues, self.goal_stds = DEQUE(), DEQUE()
        self.kl_losses, self.recon_losses = DEQUE(), DEQUE()
        self.actor_mues, self.actor_stds = DEQUE(), DEQUE()

        if expert_data_path is not None:
            self.replay_buffer = HindsightBuffer(
                expert_data_path=expert_data_path,
                observation_space=env.observation_space,
                action_space=env.action_space,
                max_traj_len=max_traj_len,
                device=self.device,
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
        self.policy = DeliG4Policy(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)
        self._convert_train_freq()

        self.actor = self.policy.actor
        self.vae = HistoryVAE(
            self.observation_space.shape[0],
            self.action_space.shape[0],
            self.vae_feature_dim,
            self.latent_dim,
            self.additional_dim,
        ).to(self.device)
        self.vae.optimizer = th.optim.Adam(
            self.vae.parameters(),
            lr=5e-4,
        )

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)
        for gradient_step in range(1):
            # Sample replay buffer
            replay_data = self.replay_buffer.goalcond_sample(batch_size, self.subtraj_len, include_current=False)

            # Define the input data by concatenating the ingradients.
            history_tensor = th.cat((replay_data.history.observations, replay_data.history.actions), dim=2)
            history_latent, history_stat = self.vae(history_tensor)

            # NOTE ---- Start: Latent vector KL-loss
            history_mu, history_log_std = history_stat
            history_std = th.exp(history_log_std)
            history_kl_loss = -0.5 * (1 + th.log(history_std.pow(2)) - history_mu.pow(2) - history_std.pow(2)).mean()

            kl_loss = history_kl_loss.mean()

            # Save logs
            self.kl_losses.append(kl_loss.item())
            self.history_mues.append(history_mu.mean().item())
            self.history_stds.append(history_std.mean().item())
            # NOTE ---- End: Latent vector KL-loss

            # NOTE ---- Start: Goal encoder-decoder reconstruction loss
            goal_recon = self.vae.decode_goal(history_tensor)
            goal_recon_loss = th.mean((goal_recon - replay_data.goal) ** 2)
            self.recon_losses.append(goal_recon_loss.item())

            vae_loss = kl_loss + goal_recon_loss
            self.vae.zero_grad()
            vae_loss.backward()
            self.vae.optimizer.step()
            # NOTE ---- End: Goal encoder-decoder reconstruction loss

            # NOTE ---- Start: entropy coefficient loss
            # Action by the current actor for the sampled state
            policy_input = th.cat((replay_data.observations, history_latent), dim=1)
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
                self.ent_coef_optimizer.step(21)
            # NOTE ---- End: entropy coefficient loss

            # NOTE ---- Start: DeliG
            # Define the input data by concatenating the ingradients.

            history_tensor = th.cat((replay_data.history.observations, replay_data.history.actions), dim=2)
            with th.set_grad_enabled(self.learn_latent):
                history_latent, _ = self.vae(history_tensor)
                print("WHAT", history_latent.requires_grad)
            policy_input = th.cat((replay_data.observations, history_latent), dim=1)
                # policy_input = th.cat((replay_data.observations, replay_data.goal, history_latent), dim=1)

            self.actor.action_log_prob(policy_input)
            # action_log_prob, actor_mu, actor_log_std \
            #     = self.actor.get_log_prob(policy_input, replay_data.actions, ret_stat = True)
            action_log_prob, actor_mu, actor_log_std \
                = self.actor.calculate_log_prob(policy_input, replay_data.actions, ret_stat=True)
            self.actor_mues.append(actor_mu.mean().item())
            self.actor_stds.append(th.exp(actor_log_std).mean().item())
            self.log_likelihood.append(action_log_prob.mean().item())

            loss = -action_log_prob.mean()
            self.actor.zero_grad()
            loss.backward()
            self.actor.optimizer.step()
            # NOTE ---- End: DeliG

        self._n_updates += gradient_steps

        self.logger.record("train/actor_mu", np.mean(self.actor_mues), exclude="tensorboard")
        self.logger.record("train/actor_stds", np.mean(self.actor_stds), exclude="tensorboard")
        self.logger.record("train/history_mu", np.mean(self.history_mues), exclude="tensorboard")
        self.logger.record("train/history_stds", np.mean(self.history_stds), exclude="tensorboard")
        self.logger.record("config/normalizing", np.mean(self.replay_buffer.normalizing), exclude="tensorboard")

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(self.ent_coefs))
        self.logger.record("train/likelihood", np.mean(self.log_likelihood))
        self.logger.record("train/recon_loss", np.mean(self.recon_losses))
        self.logger.record("train/kl_loss", np.mean(self.kl_losses))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "DeliG4",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(DeliG4, self).learn(
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
        return super(DeliG4, self)._excluded_save_params() + ["actor"]

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

    def set_training_mode(self, bool):
        self.policy.set_training_mode(bool)
        if bool:
            self.vae.train()
        else:
            self.vae.eval()
