from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.td3.policies import TD3Policy


class MIN(OffPolicyAlgorithm):
    """
    Minimalist approach for offline reinforcement learning.
    The algorithm based on TD3.

    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

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
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
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
        policy: Union[str, Type[TD3Policy]],
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
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,

        without_exploration: bool = False,
        gumbel_ensemble: bool = False,
        gumbel_temperature: float = 0.5,
        alpha: float = 2.5,
    ):

        super(MIN, self).__init__(
            policy,
            env,
            TD3Policy,
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

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        self.behavior_cloning_criterion = th.nn.MSELoss()
        self.alpha = alpha

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(MIN, self)._setup_model()
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
        bc_losses = []
        # for _ in range(gradient_steps):
        self._n_updates += 1
        # Sample replay buffer
        replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
        with th.no_grad():
            # Select action according to policy and add clipped noise
            noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

            # Compute the next Q-values: min over all critics targets
            next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
            if self.gumbel_ensemble:
                gumbel_coefs = self.get_gumbel_coefs(next_q_values, inverse_proportion=True)
                next_q_values = th.sum(next_q_values * gumbel_coefs, dim=1, keepdim=True)
            else:
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)

            target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

        # Get current Q-values estimates for each critic network
        current_q_values = self.critic(replay_data.observations, replay_data.actions)

        # Compute critic loss
        critic_loss = sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
        critic_losses.append(critic_loss.item())

        # Optimize the critics

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Delayed policy updates
        if self._n_updates % self.policy_delay == 0:
            # Compute actor loss

            # 1-1. Compute Q-network (critic) loss
            pi = self.actor(replay_data.observations)
            q_values = self.critic.q1_forward(replay_data.observations, pi)

            # 1-2. Compute lambda, a coefficient of q-net loss
            coef_lambda = self.alpha / (q_values.abs().mean().detach())

            # 2. Behavior cloning loss
            bc_loss = self.behavior_cloning_criterion(pi, replay_data.actions)
            bc_losses.append(bc_loss.item())

            # 3. Compute the minimalist approach loss
            actor_loss = -(coef_lambda * q_values.mean() - bc_loss)

            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(bc_losses) > 0:
            self.logger.record("train/bc_loss", np.mean(bc_losses))
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "MIN",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(MIN, self).learn(
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
        return super(MIN, self)._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []


class Td3CqlBc(OffPolicyAlgorithm):
    """
    Minimalist approach for offline reinforcement learning.
    The algorithm based on TD3.

    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

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
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
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
        policy: Union[str, Type[TD3Policy]],
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
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        # Added
        num_randoms: int = 10,
        without_exploration: bool = False,
        gumbel_ensemble: bool = False,
        gumbel_temperature: float = 0.5,
        alpha: float = 2.5,
        cql_coef: Union[float, str] = "auto",
        conservative_weight: float = 1.0,
        lagrange_thresh: float = 10.0,
    ):

        super(Td3CqlBc, self).__init__(
            policy,
            env,
            TD3Policy,
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

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        self.behavior_cloning_criterion = th.nn.MSELoss()
        self.cql_coef = cql_coef
        self.conservative_weight = conservative_weight
        self.lagrange_thresh = lagrange_thresh
        self.alpha = alpha
        self.num_randoms = num_randoms

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(Td3CqlBc, self)._setup_model()
        self._create_aliases()

        # The alpha coefficient of CQL can be learned automatically
        if isinstance(self.cql_coef, str) and self.cql_coef.startswith("auto"):
            init_alpha = 1.0
            self.log_cql_coef = th.log(th.ones(1, device=self.device) * init_alpha).requires_grad_(True)
            self.cql_coef_optimizer = th.optim.Adam([self.log_cql_coef], lr=self.lr_schedule(1))

        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.cql_coef_tensor = th.tensor(float(self.cql_coef)).to(self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def _get_conservative_loss(self, replay_data, current_q_values: th.Tensor) -> th.Tensor:
        """
        This is only for the CQL implementation.
        """
        # Parsing the size
        batch_size, num_actions = replay_data.actions.size()  # [batch_size, num_actions]

        # Expand the observation by num_randoms. Note that this is needed
        # to compute the equation given in appendix F of the CQL paper.
        observations = th.repeat_interleave(replay_data.observations, repeats=self.num_randoms, dim=0)  # [batch_size * num_randoms, observation_dim]

        random_actions = th.rand((batch_size, self.num_randoms, num_actions), requires_grad=True).to(self.device) * 2 - 1
        # Expand the size of observation to meet the dimension of num_randoms.
        # Get EACH values of q-networks. Note that this is based on double DQN.
        # Compute Q(s, a_i) where a_i follows the uniform distribution.

        random_action_q_values = list(self.critic.cql_forward(observations, random_actions))

        num_critics = len(random_action_q_values)
        random_density = th.log(th.tensor(0.5 ** num_actions))

        cat_input = [[qj_rand_q_val - random_density]
                 for qj_rand_q_val in random_action_q_values]

        q_cats = [th.cat(x, dim=1) for x in cat_input]      # Length = n_critics

        min_q_losses = [th.logsumexp(q_cats[j], dim=1).mean() - current_q_values[j].mean() for j in range(num_critics)]
        conservative_loss = sum(min_q_losses) / num_critics

        if self.conservative_weight > 0:
            return self.conservative_weight * conservative_loss
        else:
            return conservative_loss


    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        actor_losses, critic_losses = [], []

        bc_losses = []
        cql_coef_losses, conservative_losses = [], []
        cql_coefs = []
        self._n_updates += 1
        # Sample replay buffer
        replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
        with th.no_grad():
            # Select action according to policy and add clipped noise
            noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

            # Compute the next Q-values: min over all critics targets
            next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
            if self.gumbel_ensemble:
                gumbel_coefs = self.get_gumbel_coefs(next_q_values, inverse_proportion=True)
                next_q_values = th.sum(next_q_values * gumbel_coefs, dim=1, keepdim=True)
            else:
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)

            target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

        # Get current Q-values estimates for each critic network
        current_q_values = self.critic(replay_data.observations, replay_data.actions)

        """
        Start: CQL
        """
        # Compute critic loss
        # We can update alpha by using dual gradient method.
        cql_coef_loss = None
        # self.cql_coef_optimizer = None
        if self.cql_coef_optimizer is not None:
            cql_coef = th.exp(self.log_cql_coef.detach())
            # no grad 안하면 gradient 섞임.
            with th.no_grad():
                alpha_conservative_loss = self._get_conservative_loss(replay_data, current_q_values)
            cql_coef_loss = -th.exp(self.log_cql_coef) * (alpha_conservative_loss - self.lagrange_thresh)
            cql_coef_losses.append(cql_coef_loss.item())
        else:
            cql_coef = self.cql_coef_tensor

        # For stability, we clamp to 0 ~ 1e6
        cql_coef = cql_coef.clamp(0, 1e6)
        cql_coefs.append(cql_coef.item())

        if cql_coef_loss is not None:
            self.cql_coef_optimizer.zero_grad()
            cql_coef_loss.backward()
            self.cql_coef_optimizer.step()

        # Compute critic loss
        original_critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
        # Compute conservative loss
        conservative_loss = self._get_conservative_loss(replay_data, current_q_values)
        conservative_losses.append(conservative_loss.item())
        """
        End: CQL
        """
        conservative_loss = cql_coef * (conservative_loss - self.lagrange_thresh)
        critic_loss = conservative_loss + original_critic_loss
        critic_losses.append(critic_loss.item())

        # Optimize the critics
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Delayed policy updates
        if self._n_updates % self.policy_delay == 0:
            # Compute actor loss

            # 1-1. Compute Q-values
            pi = self.actor(replay_data.observations)
            q_values = self.critic.q1_forward(replay_data.observations, pi)
            # 1-2. Compute lambda, a coefficient of q-net loss
            coef_lambda = self.alpha / q_values.abs().mean().detach()

            """
            Start: Behavior Cloning (BC)
            """
            # 2. Behavior cloning loss
            bc_loss = self.behavior_cloning_criterion(pi, replay_data.actions)
            bc_losses.append(bc_loss.item())

            """
            End: Behavior Cloning (BC)
            """
            # 3. Compute the minimalist approach loss
            actor_loss = -(coef_lambda * q_values.mean() - bc_loss)
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(bc_losses) > 0:
            self.logger.record("train/bc_loss", np.mean(bc_losses))
        self.logger.record("train/cql_coef_loss", np.mean(cql_coef_losses))
        self.logger.record("train/conservative_loss", np.mean(conservative_losses))
        self.logger.record("train/cql_coef", np.mean(cql_coefs))
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "Td3CqlBc",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(Td3CqlBc, self).learn(
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
        return super(Td3CqlBc, self)._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []