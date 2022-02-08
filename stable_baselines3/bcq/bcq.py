from typing import Any, Dict, Optional, Tuple, Type, Union, List

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.bcq.policies import BCQPolicy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update


class BCQ(OffPolicyAlgorithm):
    """
    Batch Constraint Q-Learning

    Deterministic Policy Gradient: http://proceedings.mlr.press/v32/silver14.pdf
    DDPG Paper: https://arxiv.org/abs/1509.02971
    Introduction to DDPG: https://spinningup.openai.com/en/latest/algorithms/ddpg.html

    Note: we treat DDPG as a special case of its successor TD3.

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
        policy: Union[str, Type[BCQPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,

        # Add for BCQ
        latent_dim: int = 100,
        perturb_clipping_range: float = 0.05,
        without_exploration: bool = True,
        gumbel_ensemble: bool = False,
        gumbel_temperature: float = -1.0,
    ):
        assert without_exploration, "BCQ only for offline reinforcement learning"
        super(BCQ, self).__init__(
            policy,
            env,
            BCQPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box),

            without_exploration=without_exploration,
            gumbel_ensemble=gumbel_ensemble,
            gumbel_temperature=gumbel_temperature
        )

        self.policy_delay = 1
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        # Add for BCQ
        self.perturb_clipping_range = perturb_clipping_range
        self.latent_dim = latent_dim

        if _init_setup_model:
            self._setup_model()

        # Add for BCQ
        self.autoencoder = self.policy.actor.autoencoder
        self.ae_optimizer = th.optim.Adam(self.autoencoder.parameters(), lr=1e-4)

    def _setup_model(self) -> None:
        super(BCQ, self)._setup_model()
        self._create_aliases()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target


    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        actor_losses, critic_losses = [], []

        autoencoder_losses = []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # Train the autoencoder of BCQ
            reconstructed_action, mean, log_std = self.autoencoder(replay_data.observations, replay_data.actions)
            std = th.exp(log_std)
            ae_kl_loss = th.log(1 / std) + (std**2 + mean**2) / 2 - 0.5
            autoencoder_loss = th.mean((reconstructed_action - replay_data.actions) ** 2) + th.mean(ae_kl_loss)

            self.autoencoder.zero_grad()
            autoencoder_loss.backward()
            self.ae_optimizer.step()
            autoencoder_losses.append(autoencoder_loss.item())
            # Autoencoder train done.

            with th.no_grad():
                tile_next_observations = th.repeat_interleave(replay_data.next_observations, repeats=10, dim=0)
                tile_next_actions = self.actor(tile_next_observations)

                next_q_values = (self.critic_target.repeated_forward(tile_next_observations, tile_next_actions, batch_size))
                next_q_values = th.cat(next_q_values, dim=2)        # [batch_size, repeat, n_qs]
                n_qs = next_q_values.size(2)
                # NOTE 2: In ensemble of Q-networks, we just catch the minimum: lambda = 1 in equation (13) of the paper
                next_q_values, _ = th.min(next_q_values, dim=2)     # [batch_size, repeat]
                next_q_values, _ = th.max(next_q_values, dim=1, keepdim=True)   # [batch_size, 1]

                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values]) / n_qs
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss following the BCQ. Autoencoder의 gradient 섞이지 않기 위해, decoded action을 따로 포워딩.
            # 괜히 시발 policy안에다가 autoencoder를 집어넣어서 이렇게 해야됨. 나중에는 Autoencoder랑 policy를 따로 인스턴스만들자.
            decoded_action = self.autoencoder.decode(replay_data.observations)
            sampled_actions = self.actor.perturb_action(replay_data.observations, decoded_action.detach())

            actor_loss = -self.critic.q1_forward(replay_data.observations, sampled_actions).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/autoencoder_loss", np.mean(autoencoder_losses))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "BCQ",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(BCQ, self).learn(
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
        return super(BCQ, self)._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []
