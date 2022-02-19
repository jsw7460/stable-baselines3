from typing import Any, Dict, List, Optional, Tuple, Type, Union
from typing import Callable

import gym
import numpy as np
import torch as th

from stable_baselines3.bear.policies import VariationalAutoEncoder
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.preprocessing import get_action_dim, get_flattened_obs_dim
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.tqc.policies import TQCPolicy


class AUGTQC(OffPolicyAlgorithm):
    """

    Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics.
    Paper: https://arxiv.org/abs/2005.04269
    This implementation uses SB3 SAC implementation as base.

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
    :param gradient_steps: How many gradient update after each step
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
    :param top_quantiles_to_drop_per_net: Number of quantiles to drop per network
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
        policy: Union[str, Type[TQCPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Callable] = 3e-4,
        buffer_size: int = 1000000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
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

        # For Offline TQC
        top_quantiles_to_drop_per_net: int = 5,
        without_exploration: bool = False,
        gumbel_ensemble: bool = False,
        gumbel_temperature: float = 0.5,
        use_uncertainty: bool = True,
        truncation: bool = False,
        min_clip: int = 0,
        uc_type: str = "epistemic",
        use_policy_penalty: bool = True,

        # For BEAR
        n_sampling: int = 10,
        mmd_sigma: float = 20.0,
        delta_conf: float = 0.1,
        warmup_step: int = 20,
        mmd_thresh: float = 0.07,
    ):

        super(AUGTQC, self).__init__(
            policy,
            env,
            TQCPolicy,
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
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box),
            support_multi_env=True,

            without_exploration=without_exploration,
            gumbel_ensemble=gumbel_ensemble,
            gumbel_temperature=gumbel_temperature,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer = None
        self.top_quantiles_to_drop_per_net = top_quantiles_to_drop_per_net
        self.use_uncertainty = use_uncertainty
        self.truncation = truncation
        self.min_clip = min_clip
        self.uc_type = uc_type

        if min_clip > 0:
            assert truncation, "Min_clip is only for truncation"

        if use_uncertainty and truncation:
            raise LookupError("Use only one of uncertainty or truncation")

        # Note: Add for BEAR
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
        self.mmd_thresh = mmd_thresh
        self.use_policy_penalty = use_policy_penalty

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(AUGTQC, self)._setup_model()
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

        # Note: Add for Bear
        init_value = 1.0
        self.log_lagrange_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
        self.lagrange_coef_optimizer = th.optim.Adam([self.log_lagrange_coef], lr=1e-3)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def get_uncertainty(self, observation: Union[np.ndarray, Dict[str, np.ndarray]], actions:th.Tensor):
        """
        Only for the evaluation.
        Calculate the uncertainty of Q(s, pi(a)) where pi is an action selected by the actor.
        Thus here, the parameter actions is an output of the actor.
        """
        observation, vectorized_env = self.policy.obs_to_tensor(observation)
        with th.no_grad():
            quantiles = self.critic_target(observation, th.tensor(actions, device=self.device))
        truncated_quantiles = quantiles[:, :, self.top_quantiles_to_drop_per_net : -self.top_quantiles_to_drop_per_net]

        aleatoric = th.mean(truncated_quantiles, dim=1)
        aleatoric = th.var(aleatoric, dim=1, keepdim=True)      # [batch_size, 1]

        epistemic = th.var(truncated_quantiles, dim=1)
        epistemic = th.mean(epistemic, dim=1, keepdim=True)     # [batch_size, 1]

        return aleatoric.squeeze().item(), epistemic.squeeze().item()

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

        variance_mins, variance_maxs = [], []
        trunc_mins, trunc_maxs, trunc_means = [], [], []

        aleatoric_uncs, epistemic_uncs = [], []

        min_q_vals, max_q_vals, mean_q_vals = [], [], []

        policy_penalty_mins, policy_penalty_maxs, policy_penalty_means = [], [], []

        # Add for Bear
        autoencoder_losses = []
        mmd_losses = []
        bc_losses = []
        lagrange_coefs = []
        lagrange_coef_losses = []

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
        self.replay_buffer.ent_coef = ent_coef.item()

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if ent_coef_loss is not None:
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()

        with th.no_grad():
            # Select action according to policy
            next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
            # Compute and cut quantiles at the next state
            # batch x nets x quantiles
            n_total_quantiles = self.critic.quantiles_total     # = n_quantiles * n_critics
            next_quantiles = self.critic_target(replay_data.next_observations, next_actions)

            # Save the statistics for the log
            _next_quantiles = next_quantiles.detach()
            _batch_q_val = th.mean(_next_quantiles, dim=(1, 2))
            _max = th.max(_batch_q_val)
            _min = th.min(_batch_q_val)
            _mean = th.mean(_batch_q_val)

            max_q_vals.append(_max.item())
            min_q_vals.append(_min.item())
            mean_q_vals.append(_mean.item())

            # NOTE: The number of quantiles to use. It will be changed according to variance of reward,
            # if use truncation mode (See if self.truncation loop)
            n_target_quantiles = n_total_quantiles - self.top_quantiles_to_drop_per_net * self.critic.n_critics
            # [batch x nets x remained quantiles].
            trun_next_quantiles = next_quantiles[:, :, self.top_quantiles_to_drop_per_net : -self.top_quantiles_to_drop_per_net]

            # n_target_quantiles = self.critic.quantiles_total
            next_quantiles = next_quantiles.reshape(batch_size, -1)
            next_quantiles, _ = th.sort(next_quantiles)
            if self.truncation:
                """
                Coefficient of bellman updqte is one
                But the number of truncation varies
                """
                uncertainty_coefs = 1.0       # No coefficient penalty if use truncation

                with th.no_grad():
                    # Aleatoric variance
                    a_quantiles_variance = th.mean(trun_next_quantiles, dim=1)
                    a_quantiles_variance = th.var(a_quantiles_variance, dim=1, keepdim=True)  # [batch_size x 1]

                    # Epistemic variance
                    e_quantiles_variance = th.var(trun_next_quantiles, dim=1)
                    e_quantiles_variance = th.mean(e_quantiles_variance, dim=1, keepdim=True)  # [batch_size x 1]

                    aleatoric_uncs.append(a_quantiles_variance.mean().item())
                    epistemic_uncs.append(e_quantiles_variance.mean().item())

                if self.uc_type == "aleatoric":
                    quantiles_variance = a_quantiles_variance
                elif self.uc_type == "epistemic":
                    quantiles_variance = e_quantiles_variance
                elif self.uc_type == "both":
                    quantiles_variance = a_quantiles_variance + e_quantiles_variance

                n_target_quantiles \
                    = th.floor(n_total_quantiles * th.exp(-quantiles_variance + 5.0)).type(th.IntTensor)
                n_target_quantiles = n_target_quantiles.to(self.device)

                # If we set min_clip very large, then it has no offline RL penalty effect
                th.clip_(n_target_quantiles, self.min_clip, 100000)
                masking = th.arange(0, n_total_quantiles, device=self.device).unsqueeze(0)
                masking = masking.expand(batch_size, n_total_quantiles)
                masking = th.where(n_target_quantiles - masking >= 0, 1, 0)

                # Drop top k quantiles to control overestimation
                next_quantiles = next_quantiles * masking
                n_trunc = th.sum(masking, dim=1, dtype=th.float32)

                next_quantiles, _ = th.sort(next_quantiles)

                trunc_max, _ = th.max(n_trunc, dim=0)
                trunc_min, _ = th.min(n_trunc, dim=0)
                trunc_mean = th.mean(n_trunc)
                trunc_maxs.append(trunc_max.item())
                trunc_mins.append(trunc_min.item())
                trunc_means.append(trunc_mean.item())

            elif self.use_uncertainty:
                raise NotImplementedError("이거 쓰지 마세요 ~!!!!!!!!!!!!!!!!!!")
                """
                No truncation, but manipulate the uncertainty coefficient to penalty the bellman operation
                """
                if self.uc_type == "epistemic":
                    quantiles_variance = th.var(trun_next_quantiles, dim=2)
                    quantiles_variance = th.mean(quantiles_variance, dim=1, keepdim=True)   # [batch_size x 1]

                elif self.uc_type == "aleatoric":
                    quantiles_variance = th.mean(trun_next_quantiles, dim=1)
                    quantiles_variance = th.var(quantiles_variance, dim=2, keepdim=True)

                var_max, _ = th.max(quantiles_variance, dim=0)
                var_min, _ = th.min(quantiles_variance, dim=0)
                variance_maxs.append(var_max.item())
                variance_mins.append(var_min.item())  # Log before the clipping
                uncertainty_coefs = (1 / quantiles_variance).clip_(0.0, 1.0)  # [batch_size, 1]

                # [batch_size, n_target_quantiles]
                next_quantiles = next_quantiles[:, : n_target_quantiles]  # n_target_quantiles 만큼만 남기겠다

            else:
                raise NotImplementedError("이거 쓰지 마세요 ~!!!!!!!!!!!!!!!!!!")
                uncertainty_coefs = 1
                # [batch_size, n_target_quantiles]
                next_quantiles = next_quantiles[:, : n_target_quantiles]  # n_target_quantiles 만큼만 남기겠다

            # td error + entropy term
            target_quantiles = next_quantiles - ent_coef * next_log_prob.reshape(-1, 1)
            target_quantiles = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_quantiles
            # Make target_quantiles broadcastable to (batch_size, n_critics, n_target_quantiles).
            target_quantiles.unsqueeze_(dim=1)

        # Get current Quantile estimates using action from the replay buffer
        current_quantiles = self.critic(replay_data.observations, replay_data.actions)

        # Compute critic loss, not summing over the quantile dimension as in the paper.
        critic_loss \
            = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=False, without_mean=True)
        critic_loss = critic_loss.view(batch_size, -1)
        critic_loss = (uncertainty_coefs * critic_loss).mean()

        critic_losses.append(critic_loss.item())

        # Optimize the critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Note: ------------- START BEAR POLICY NETWORK UPDATE WITH PENALTY
        # -------------------------------
        # p_: policy 부분에 대한 변수 naming
        # -------------------------------

        tile_current_observations = th.repeat_interleave(replay_data.observations, repeats=10, dim=0)
        tile_current_actions, raw_current_actions = self.actor.forward_with_pretanh(tile_current_observations)

        # NOTE: 여기서 critic_target이라고 하는 짓을 하면 안된다.
        # NOTE: 왜냐면 q-value높이는 쪽으로 actor를 학습 시키는데 target net으로 넣으면 gradient 사라짐 !!
        p_current_quantiles = self.critic.forward(replay_data.observations, actions_pi)

        policy_penalty = 1.0
        if self.use_policy_penalty:
            with th.no_grad():
                # [batch_size, n_qs, remained_quantiles]
                p_trunc_current_quantiles = p_current_quantiles[:, :, self.top_quantiles_to_drop_per_net : -self.top_quantiles_to_drop_per_net]
                p_trunc_current_quantiles = p_trunc_current_quantiles.reshape(batch_size, -1)
                p_trunc_current_quantiles, _ = th.sort(p_trunc_current_quantiles)

                policy_penalty = 1 / th.var(p_trunc_current_quantiles, dim=1, keepdim=True)       # [batch_size, 1]
                policy_penalty.clamp_(0.5, 1.0)

            pp_min, _ = th.min(policy_penalty, dim=0)
            pp_max, _ = th.max(policy_penalty, dim=0)

            policy_penalty_mins.append(pp_min.item())
            policy_penalty_maxs.append(pp_max.item())
            policy_penalty_means.append(policy_penalty.mean().item())

        current_q_values = th.mean(p_current_quantiles, dim=2)      # [batch_size, n_qs]
        if self.gumbel_ensemble:
            gumbel_coefs = self.get_gumbel_coefs(current_q_values, inverse_proportion=True)
            min_qf_pi = gumbel_coefs * current_q_values
        else:
            min_qf_pi, _ = th.min(current_q_values, dim=1)      # [batch_size]

        # current_q_values = current_q_values[:, :self.top_quantiles_to_drop_per_net]
        actor_loss = policy_penalty * (ent_coef * log_prob - min_qf_pi)       # SAC style policy update

        # Compute mmd loss
        vae_actions, raw_vae_actions \
            = self.autoencoder.decode(tile_current_observations, device=self.device, return_pretanh=True)

        vae_actions = vae_actions.view(batch_size, 10, -1)
        # raw_vae_actions = raw_vae_actions.view(batch_size, 10, -1)

        policy_actions = tile_current_actions.view(batch_size, 10, -1)
        # raw_policy_actions = raw_current_actions.view(batch_size, 10, -1)
        # mmd_loss = self.laplacian_mmd_loss(raw_vae_actions.detach(), raw_policy_actions)
        mmd_loss = self.laplacian_mmd_loss(vae_actions.detach(), policy_actions)
        mmd_losses.append(mmd_loss.item())
        # End mmd loss
        log_lagrange_coef = self.log_lagrange_coef

        if self.offline_round_step < self.warmup_step:
            actor_loss = 100.0 * mmd_loss  # BEAR의 웜업
        else:
            total_mmd_loss = th.exp(log_lagrange_coef) * (mmd_loss - self.mmd_thresh)
            actor_loss = actor_loss.mean() + total_mmd_loss

        actor_losses.append(actor_loss.item())

        # Optimize the actor
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Optimize the lagrange coefficient
        lagrange_coef = th.exp(self.log_lagrange_coef.detach())
        lagrange_coef_loss = None
        if self.offline_round_step > self.warmup_step:
            lagrange_coef_loss = -th.exp(self.log_lagrange_coef) * (mmd_loss.detach() - self.mmd_thresh)
            lagrange_coef_losses.append(lagrange_coef_loss.item())
            self.log_lagrange_coef.data.clamp_(-5.0, 10.0)

        lagrange_coefs.append(lagrange_coef.item())

        if lagrange_coef_loss is not None:
            self.lagrange_coef_optimizer.zero_grad()
            lagrange_coef_loss.backward()
            self.lagrange_coef_optimizer.step()

        # Update target networks
        polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

        # TQC
        self.logger.record("config/n_trunc_quantiles", self.top_quantiles_to_drop_per_net)
        self.logger.record("config/quantile_clip", self.min_clip)
        self.logger.record("config/uc_type", self.uc_type)

        if self.use_uncertainty:
            self.logger.record("config/uncertainty", 1)
            self.logger.record("train/max_var", np.mean(variance_maxs))
            self.logger.record("train/min_var", np.mean(variance_mins))

        if self.truncation:
            self.logger.record("config/truncation", 1)
            self.logger.record("train/max_trunc", np.mean(trunc_maxs))
            self.logger.record("train/min_trunc", np.mean(trunc_mins))
            self.logger.record("train/mean_trunc", np.mean(trunc_means))
            if len(aleatoric_uncs) > 0:
                self.logger.record("train/aleatoric_unc", np.mean(aleatoric_uncs))
            if len(epistemic_uncs) > 0:
                self.logger.record("train/epistemic_unc", np.mean(epistemic_uncs))
        if self.use_policy_penalty:
            self.logger.record("config/use_pol_pen", "True")
            self.logger.record("train/pol_pen_min", np.mean(policy_penalty_mins))
            self.logger.record("train/pol_pen_max", np.mean(policy_penalty_maxs))
            self.logger.record("train/pol_pen_mean", np.mean(policy_penalty_means))

        self.logger.record("train/Q-val-min", np.mean(min_q_vals))
        self.logger.record("train/Q-val-max", np.mean(max_q_vals))
        self.logger.record("train/Q-val-mean", np.mean(mean_q_vals))

        # Bear
        self.logger.record("train/autoencoder_loss", np.mean(autoencoder_losses))
        if len(mmd_losses) > 0:
            self.logger.record("train/mmd_thresh", self.mmd_thresh)
            self.logger.record("train/mmd_loss", np.mean(mmd_losses))
        if len(bc_losses) > 0:
            self.logger.record("train/bc_loss", np.mean(bc_losses))
        self.logger.record("train/lagrange_coef", np.mean(lagrange_coefs))
        if len(lagrange_coef_losses) > 0:
            self.logger.record("train/lagrange_loss", np.mean(lagrange_coef_losses))
        self.logger.record("config/warmup_step", self.warmup_step)
        self.logger.record("config/n_quantiles", self.critic.n_quantiles)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "AUGTQC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(AUGTQC, self).learn(
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
        # Exclude aliases
        return super(AUGTQC, self)._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables