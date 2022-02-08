from typing import Any, Dict, Optional, Tuple, Type, Union, List

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.aubcq.policies import VariationalAutoEncoder, NextStateAutoEncoder, SACPolicy
from stable_baselines3.common.preprocessing import get_action_dim, get_flattened_obs_dim
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update


class SACAUBCQ(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
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

        # add for AUBCQ
        without_exploration: bool = False,
        gumbel_ensemble: bool = False,
        gumbel_temperature: float = 0.5,

        latent_dim: int = 100,
        aug_critic_coef: float = 0.1,
        warmup_step: int = 30,
    ):
        assert without_exploration, "BCQ only for offline reinforcement learning"
        super(SACAUBCQ, self).__init__(
            policy,
            env,
            SACPolicy,
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
            without_exploration=without_exploration,
            gumbel_ensemble=gumbel_ensemble,
            gumbel_temperature=gumbel_temperature
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer = None

        if _init_setup_model:
            self._setup_model()

            # Add for AUBCQ
            state_dim = get_flattened_obs_dim(self.observation_space)
            max_state = self.env.observation_space.high[0]

            action_dim = get_action_dim(self.action_space)
            max_action = self.env.action_space.high[0]

            self.action_autoencoder = VariationalAutoEncoder(state_dim, action_dim, latent_dim, max_action).to(self.device)
            self.nextstate_autoencoder = NextStateAutoEncoder(state_dim, action_dim, latent_dim, max_state).to(self.device)

            self.action_autoencoder_optimizer = th.optim.Adam(self.action_autoencoder.parameters(), lr=5e-4)
            self.nextstate_autoencoder_optimizer = th.optim.Adam(self.nextstate_autoencoder.parameters(), lr=5e-4)
            self.warmup_step = warmup_step
            self.aug_critic_coef = aug_critic_coef

    def _setup_model(self) -> None:
        super(SACAUBCQ, self)._setup_model()
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
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        dataset_critic_losses = []
        augmented_critic_losses = []
        action_autoencoder_losses = []
        nextstate_autoencoder_losses = []
        overestimation_ratios, overest_ratio_mins, overest_ratio_maxs = [], [], []
        data_bc_losses, ae_bc_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())
            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # Start: Autoencoder train 1. Action generating autoencoder traning
            reconstructed_action, mean, log_std \
                = self.action_autoencoder(replay_data.observations, replay_data.actions)

            std = th.exp(log_std)
            ae_kl_loss = th.log(1 / std) + (std ** 2 + mean ** 2) / 2 - 0.5
            action_autoencoder_loss = th.mean((reconstructed_action - replay_data.actions) ** 2) + th.mean(
                ae_kl_loss)

            self.action_autoencoder.zero_grad()
            action_autoencoder_loss.backward()
            self.action_autoencoder_optimizer.step()
            action_autoencoder_losses.append(action_autoencoder_loss.item())
            # End: Autoencoder train 1

            # Autoencoder train 2. Next state generating autoencoder training
            reconstructed_next_state, mean, log_std \
                = self.nextstate_autoencoder(
                replay_data.observations,
                replay_data.actions,
                replay_data.next_observations,
            )
            std = th.exp(log_std)
            ae_kl_loss = th.log(1 / std) + (std ** 2 + mean ** 2)
            state_autoencoder_loss \
                = th.mean((reconstructed_next_state - replay_data.next_observations) ** 2) + th.mean(ae_kl_loss)

            # TODO: observation space를 normalize 해서 [0, 1]사이 값으로 만들어줘도 나쁘지 않을 것으로 보임.
            self.nextstate_autoencoder.zero_grad()
            state_autoencoder_loss.backward()
            self.nextstate_autoencoder_optimizer.step()
            nextstate_autoencoder_losses.append(state_autoencoder_loss.item())

            with th.no_grad():
                tile_next_observations = th.repeat_interleave(replay_data.next_observations, repeats=10, dim=0)
                tile_next_actions = self.actor(tile_next_observations)

                next_q_values = (self.critic_target.repeated_forward(tile_next_observations, tile_next_actions, batch_size))
                next_q_values = th.cat(next_q_values, dim=2)  # [batch_size, repeat, n_qs]
                n_qs = next_q_values.size(2)
                # NOTE 2: In ensemble of Q-networks, we just catch the minimum: lambda = 1 in equation (13) of the paper
                next_q_values, min_inds = th.min(next_q_values, dim=2)  # q_val: [batch_size, repeat]
                next_q_values, max_inds = th.max(next_q_values, dim=1, keepdim=True)  # q_val: [batch_size, 1]

                reshaped_tile_next_actions = tile_next_actions.view(batch_size, 10, -1)
                max_inds = max_inds.unsqueeze(2).repeat(1, 1, reshaped_tile_next_actions.size(2))

                # Q-function이 고른 action들임
                selected_max_action = th.gather(reshaped_tile_next_actions, 1, max_inds).squeeze()  # [batch_size, action_dim]

                overestimated_ratio = th.mean(1 / th.exp(0.25 * ((reconstructed_action - selected_max_action) ** 2)), dim=1, keepdim=True)  # [batch_size, 1]
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            min_ratio, _ = th.min(overestimated_ratio, dim=0)
            max_ratio, _ = th.max(overestimated_ratio, dim=0)
            overestimation_ratios.append(th.mean(overestimated_ratio).item())
            overest_ratio_mins.append(min_ratio.item())
            overest_ratio_maxs.append(max_ratio.item())

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = overestimated_ratio * sum([(current_q - target_q_values) ** 2 for current_q in current_q_values]) / n_qs
            critic_loss = critic_loss.mean()

            dataset_critic_losses.append(critic_loss.item())
            # End: Q-learning with dataset

            # Start: Q-learning with augmented dataset
            # NOTE: This only after warmup step
            if self.offline_round_step >= self.warmup_step:
                with th.no_grad():
                    aug_next_observations, _, _ = self.nextstate_autoencoder(
                        replay_data.observations,
                        replay_data.actions,
                        replay_data.next_observations,
                    )
                    aug_tile_next_observations = th.repeat_interleave(aug_next_observations, repeats=10, dim=0)

                    aug_tile_next_actions = self.action_autoencoder.decode(
                        aug_tile_next_observations,
                        device=self.device
                    )

                    # NOTE: 여기 CQL같은걸 넣어봐도 될 듯
                    aug_next_q_values \
                        = (self.critic_target.repeated_forward(
                        aug_tile_next_observations,
                        aug_tile_next_actions,
                        batch_size
                        )
                    )

                    aug_next_q_values = th.cat(aug_next_q_values, dim=2)  # [batch_size, repeat, n_qs]
                    # Minimum over the q-networks
                    aug_next_q_values, _ = th.min(aug_next_q_values, dim=2)  # [batch_size, repeat]
                    # Maximum over the sampled actions
                    aug_next_q_values, _ = th.min(aug_next_q_values, dim=1, keepdim=True)  # [batch_size, 1]

                    aug_target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * aug_next_q_values

                # TODO: 여기에서 aug_current_q_values를 뽑는 방법을 dataset에서 뽑지 말고 autoencoder에서 뽑아도 문제는 없을 것으로 보임
                current_q_values = self.critic(replay_data.observations, replay_data.actions)

                # Compute augmented critic loss
                aug_critic_loss = self.aug_critic_coef \
                                  * sum([F.mse_loss(current_q, aug_target_q_values) for current_q in current_q_values])
                aug_critic_loss = aug_critic_loss / n_qs    # mean
                augmented_critic_losses.append(aug_critic_loss.item())

                critic_loss += aug_critic_loss

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            q_values_pi = th.cat(self.critic.forward(replay_data.observations, actions_pi), dim=1)

            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            # Note: coef_lambda: minimalist approach style coefficient setting.
            coef_lambda = 2.5 / min_qf_pi.abs().mean().detach()
            actor_loss = (ent_coef * log_prob - coef_lambda * min_qf_pi.detach()).mean()

            data_bc_loss = th.mean((actions_pi - replay_data.actions) ** 2)
            ae_bc_loss = th.mean((actions_pi - reconstructed_action.detach()) ** 2)
            # actor_loss = actor_loss + data_bc_loss + ae_bc_loss
            actor_loss = actor_loss + data_bc_loss

            data_bc_losses.append(data_bc_loss.item())
            ae_bc_losses.append(ae_bc_loss.item())
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/data_bc_loss", np.mean(data_bc_losses))
            self.logger.record("train/ae_bc_loss", np.mean(ae_bc_losses))

        self.logger.record("train/dataset_critic_loss", np.mean(dataset_critic_losses))
        if len(augmented_critic_losses) > 0:
            self.logger.record("train/augmented_critic_loss", np.mean(augmented_critic_losses))

        self.logger.record("train/action_ae_loss", np.mean(action_autoencoder_losses))
        self.logger.record("train/next_state_ae_loss", np.mean(nextstate_autoencoder_losses))
        self.logger.record("train/overestimation_mean", np.mean(overestimation_ratios))
        self.logger.record("train/overestimation_max", np.mean(overest_ratio_maxs))
        self.logger.record("train/overestimation_min", np.mean(overest_ratio_mins))
        self.logger.record("config/augmetned_critic_coef", np.mean(self.aug_critic_coef))
        self.logger.record("config/warmup_step", np.mean(self.warmup_step))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "SACAUBCQ",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(SACAUBCQ, self).learn(
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
        return super(SACAUBCQ, self)._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
