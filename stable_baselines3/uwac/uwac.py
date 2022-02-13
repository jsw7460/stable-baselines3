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
from stable_baselines3.bear.policies import VariationalAutoEncoder
from stable_baselines3.uwac.policies import UwacPolicy
from stable_baselines3.common.preprocessing import get_action_dim, get_flattened_obs_dim


class UWAC(OffPolicyAlgorithm):
    """
    BEAR (Bootstrapping Error Accumulation Reduction)

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
        policy: Union[str, Type[UwacPolicy]],
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

        # Added for UWAC
        without_exploration: bool = False,
        gumbel_ensemble: bool = False,
        gumbel_temperature: float = 0.5,
        lagrange_coef: Union[str, float] = "auto",
        lagrange_thresh: float = 0.07,
        n_sampling: int = 10,
        mmd_sigma: float = 20.0,
        delta_conf: float = 0.1,
        warmup_step: int = 40,
        dropout: float = 0.5,
        uwac_beta = 0.5,
    ):

        super(UWAC, self).__init__(
            policy,
            env,
            UwacPolicy,
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
        self.dropout = dropout

        if _init_setup_model:
            self._setup_model()

        # Add for BEAR
        state_dim = get_flattened_obs_dim(self.observation_space)
        action_dim = get_action_dim(self.action_space)
        # Autoencoder: used to select the next state action
        self.autoencoder = VariationalAutoEncoder(
            state_dim,
            action_dim,
            100,
            self.action_space.high[0],
            self.device
        ).to(self.device)

        self.ae_optimizer = th.optim.Adam(self.autoencoder.parameters(), lr=1e-4)
        self.n_sampling = n_sampling
        self.mmd_sigma = mmd_sigma
        self.delta_conf = delta_conf
        self.warmup_step = warmup_step
        self.beta = uwac_beta

        self.lagrange_thresh = lagrange_thresh
        self.lagrange_coef = lagrange_coef
        self.lagrange_coef_optimizer = None

        if isinstance(self.lagrange_coef, str) and self.lagrange_coef.startswith("auto"):
            # Default initial value of lagrange coef when learned
            init_value = 1.0
            self.log_lagrange_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.lagrange_coef_optimizer = th.optim.Adam([self.log_lagrange_coef], lr=1e-3)
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.lagrange_coef_tensor = th.tensor(float(self.lagrange_coef)).to(self.device)
            self.log_lagrange_coef = self.lagrange_coef_tensor.requires_grad_(False)

    def _setup_model(self) -> None:
        super(UWAC, self)._setup_model()
        self._create_aliases()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def gaussian_mmd_loss(self, sample_1: th.Tensor, sample_2: th.Tensor) -> th.Tensor:
        """
        sample_1: [batch, n, dim]
        sample_2: [batch, m, dim]       # In general, n = m and where n, m: number of samplings to compute mmd
        """
        xx = sample_1.unsqueeze(2) - sample_1.unsqueeze(1)      # [batch, n, n, dim]
        xy = sample_1.unsqueeze(2) - sample_2.unsqueeze(1)      # [batch, n, m, dim]
        yy = sample_2.unsqueeze(2) - sample_2.unsqueeze(1)      # [batch, m, m, dim]

        k_xx = th.exp(-(xx ** 2) / (2 * self.mmd_sigma))
        k_xy = th.exp(-(xy ** 2) / (2 * self.mmd_sigma))
        k_yy = th.exp(-(yy ** 2) / (2 * self.mmd_sigma))

        return k_xx.mean() - 2 * k_xy.mean() + k_yy.mean()

    def laplacian_mmd_loss(self, sample_1: th.Tensor, sample_2: th.Tensor) -> th.Tensor:
        """
        sample_1: [batch, n, dim]
        sample_2: [batch, m, dim]       # In general, n = m and where n, m: number of samplings to compute mmd
        """
        diff_x_x = sample_1.unsqueeze(2) - sample_1.unsqueeze(1)  # B x N x N x d
        diff_x_x = th.mean((-(diff_x_x.pow(2)).sum(-1) / (2.0 * self.mmd_sigma)).exp(), dim=(1, 2))

        diff_x_y = sample_1.unsqueeze(2) - sample_2.unsqueeze(1)
        diff_x_y = th.mean((-(diff_x_y.pow(2)).sum(-1) / (2.0 * self.mmd_sigma)).exp(), dim=(1, 2))

        diff_y_y = sample_2.unsqueeze(2) - sample_2.unsqueeze(1)  # B x N x N x d
        diff_y_y = th.mean((-(diff_y_y.pow(2)).sum(-1) / (2.0 * self.mmd_sigma)).exp(), dim=(1, 2))

        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss.mean()

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        actor_losses, critic_losses = [], []

        autoencoder_losses = []
        mmd_losses = []
        lagrange_coefs = []

        # Added for UWAC
        qnet_vars_current, qnet_vars_next = [], []
        if self.lagrange_coef_optimizer is not None:
            lagrange_coef_losses = []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # Start: train autoencoder
            reconstructed_action, mean, log_std = self.autoencoder(replay_data.observations, replay_data.actions)
            std = th.exp(log_std)
            ae_kl_loss = th.log(1 / std) + (std ** 2 + mean ** 2) / 2 - 0.5
            autoencoder_loss = th.mean((reconstructed_action - replay_data.actions) ** 2) + th.mean(ae_kl_loss)

            self.autoencoder.zero_grad()
            autoencoder_loss.backward()
            self.ae_optimizer.step()
            autoencoder_losses.append(autoencoder_loss.item())
            # End: train autoencoder

            with th.no_grad():
                # Select action according to policy and add clipped noise
                tile_next_observations = th.repeat_interleave(replay_data.next_observations, repeats=100, dim=0)
                tile_next_actions = self.actor(tile_next_observations)

                noise = tile_next_actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)

                tile_next_actions = tile_next_actions + noise

                next_q_values = self.critic_target.repeated_forward(
                    tile_next_observations,
                    tile_next_actions,
                    batch_size
                )
                next_q_values = th.cat(next_q_values, dim=2)        # [batch_size, n_sampling, num_qs]
                n_qs = next_q_values.size(2)    # The number of q-networks

                # To update via UWAC, calculate the variance
                # NOTE: 여기는 "Next state"에 대한 variance를 이용해서 uwac weight를 구한다.
                # NOTE: 좀 이따가는 "Current state"에 대한 variance를 이용한다.
                # NOTE: 구하는 과정이 왜 variance를 더하는지는 모르겠으나 우선 UWAC의 코드를 따라하자 ....
                qnet_var_next = th.var(next_q_values, dim=1)    # [batch_size, num_qs]
                qnet_vars_next.append(qnet_var_next.mean().item())

                next_q_values, _ = th.min(next_q_values, dim=2)     # minimum over q_networks
                next_q_values, _ = th.max(next_q_values, dim=1, keepdim=True)       # maximum over samplings.   [batch_size, 1]
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Compute the weight of UWAC
            qnet_var_sum_next = th.sum(qnet_var_next, dim=1, keepdim=True)
            uwac_weight_next = th.exp(-self.beta * qnet_var_sum_next)
            uwac_weight_next.clamp_(0.0, 1.0)

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(
                [(uwac_weight_next * (current_q - target_q_values)) ** 2 for current_q in current_q_values]
            )
            critic_loss = critic_loss / n_qs
            critic_loss = th.mean(critic_loss)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates를 UWAC 논문에서는 하지를 않았다
            # Compute actor loss
            tile_current_observations = th.repeat_interleave(replay_data.observations, repeats=1000, dim=0)
            tile_current_actions = self.actor(tile_current_observations)

            noise = tile_current_actions.clone().data.normal_(0, self.target_policy_noise)
            noise.clamp_(-self.target_noise_clip, self.target_noise_clip)

            # Add the noise term
            # Note: Bear의 저자들은 policy를 tanh + Gaussian으로 하였으나 여기서는 TD3의 policy가 deterministic network이다.
            # 따라서, 우리는 여기서 직접 noise를 더해준다. 이게 큰 문제가 될 것 같지는 않음.
            tile_current_actions = tile_current_actions + noise

            current_q_values = self.critic.repeated_forward(
                tile_current_observations,
                tile_current_actions,
                batch_size
            )
            current_q_values = th.cat(current_q_values, dim=2)      # current_q_values: [batch_size, n_repeates, n_qs]
            # NOTE: 여기서는 current state에 대한 정보로 Uwac weight를 구한다
            qnet_var_current = th.var(current_q_values, dim=1)      # [batch_size, n_qs]
            qnet_vars_current.append(qnet_var_current.mean().item())

            current_q_values = th.mean(current_q_values, dim=1)
            current_q_values, _ = th.min(current_q_values, dim=1, keepdim=True)     # [batch_size, 1]

            qnet_var_sum_current = th.sum(qnet_var_current, dim=1, keepdim=True)
            uwac_weight_current = th.exp(-self.beta * qnet_var_sum_current).detach()
            uwac_weight_current.clamp_(0.0, 1.0)

            # Compute mmd loss
            vae_actions = self.autoencoder.decode(tile_current_observations, device=self.device).view(batch_size, 10, -1)
            policy_actions = tile_current_actions.view(batch_size, 10, -1)

            mmd_loss = self.laplacian_mmd_loss(vae_actions.detach(), policy_actions)
            mmd_losses.append(mmd_loss.item())

            log_lagrange_coef = self.log_lagrange_coef \
                if self.lagrange_coef_optimizer is not None \
                else self.lagrange_coef_tensor

            if self.offline_round_step < self.warmup_step:
                actor_loss = 100.0 * (mmd_loss - self.lagrange_thresh)
            else:
                actor_loss = -th.mean(current_q_values * uwac_weight_current) \
                             + th.exp(log_lagrange_coef) * (mmd_loss - self.lagrange_thresh)

            actor_loss = actor_loss.mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Optimize the lagrange coefficient if use auto regression
            if self.offline_round_step > self.warmup_step:
                lagrange_coef_loss = None
                if self.lagrange_coef_optimizer is not None:
                    lagrange_coef = th.exp(self.log_lagrange_coef.detach())
                    lagrange_coef_loss = -th.exp(self.log_lagrange_coef) * (mmd_loss.detach() - self.lagrange_thresh)
                    lagrange_coef_losses.append(lagrange_coef_loss.item())
                    self.log_lagrange_coef.data.clamp_(-5.0, 10.0)
                else:
                    lagrange_coef = th.exp(log_lagrange_coef).detach()

                lagrange_coefs.append(lagrange_coef.item())

                if lagrange_coef_loss is not None:
                    self.lagrange_coef_optimizer.zero_grad()
                    lagrange_coef_loss.backward()
                    self.lagrange_coef_optimizer.step()

            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/autoencoder_loss", np.mean(autoencoder_losses))
        if len(mmd_losses) > 0:
            self.logger.record("train/mmd_loss", np.mean(mmd_losses))
        if len(lagrange_coefs) > 0:
            self.logger.record("train/lagrange_coef", np.mean(lagrange_coefs))
        if len(lagrange_coef_losses) > 0 :
            self.logger.record("train/lagrange_loss", np.mean(lagrange_coef_losses))

        # UWAC
        self.logger.record("config/dropout", self.dropout)
        self.logger.record("config/mmd_thresh", self.lagrange_thresh)
        self.logger.record("train/var_current", np.mean(qnet_vars_current))
        self.logger.record("train/var_next", np.mean(qnet_vars_next))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "UWAC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(UWAC, self).learn(
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
        return super(UWAC, self)._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []
