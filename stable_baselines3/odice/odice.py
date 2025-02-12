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
from stable_baselines3.odice.policies import DonskerMinimizer
from stable_baselines3.odice.policies import get_donsker_loss, get_approx_kl
from stable_baselines3.sac.policies import SACPolicy

G_USE_PRIOR = True
G_PRIOR_VARIANCE = 1

G_RAND_REG = True
WARMUP = 80


class SACOdice(OffPolicyAlgorithm):
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

        without_exploration: bool = False,
        gumbel_ensemble: bool = False,
        gumbel_temperature: float = 0.5,
        warmup: int = 20,
    ):

        super(SACOdice, self).__init__(
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

        self.warmup = warmup
        self.donsker = DonskerMinimizer(self.observation_space).to(self.device)
        self.donsker_optim = th.optim.Adam(self.donsker.parameters(), lr=1e-3)

        self.behavior_policy = self.policy.make_actor()
        self.behavior_policy.optimizer = th.optim.Adam(self.behavior_policy.parameters(), lr=1e-4)

        self.initial_copy = True

    def _setup_model(self) -> None:
        super(SACOdice, self)._setup_model()
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
        if self.initial_copy:
            polyak_update(self.behavior_policy.parameters(), self.actor.parameters(), tau=1.0)
            self.initial_copy = False

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

        behavior_bc_losses = []
        target_bc_losses = []
        donsker_net_losses = []
        approx_kl_means = []

        mean_t_vars, mean_b_vars = [], []
        max_t_vars, max_b_vars = [], []
        mean_rnds, max_rnds = [], []

        for gradient_step in range(gradient_steps):
            # th.autograd.set_detect_anomaly(True)
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # NOTE: Approximate the behavior policy
            if self.offline_round_step < WARMUP:
                polyak_update(self.actor.parameters(), self.behavior_policy.parameters(), tau=1.0)

            else:
                bc_loss = -th.mean(self.behavior_policy.get_log_prob(replay_data.observations, replay_data.actions))
                behavior_bc_losses.append(bc_loss.item())

                if G_USE_PRIOR:
                    b_mean, b_log_std, _ = self.behavior_policy.get_action_dist_params(replay_data.observations)

                    b_var = th.exp(b_log_std) ** 2
                    prod_b_var = th.prod(b_var, dim=1)
                    # KL-divergence between the two Gaussians
                    # Since we have prior knowledge on the variance, so regulate only for the variance.
                    var_loss = th.log((G_PRIOR_VARIANCE ** self.action_dim) / prod_b_var) \
                               - self.action_dim + th.sum(b_var / (G_PRIOR_VARIANCE ** self.action_dim), dim=1)

                    bc_loss = bc_loss + th.mean(var_loss)

                self.behavior_policy.optimizer.zero_grad()
                bc_loss.backward()
                self.behavior_policy.optimizer.step()

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

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)

                if self.gumbel_ensemble:
                    gumbel_coefs = self.get_gumbel_coefs(next_q_values, inverse_proportion=True)
                    next_q_values = th.sum(next_q_values * gumbel_coefs, dim=1, keepdim=True)
                else:
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            if G_RAND_REG:
                random_actions = th.randn_like(replay_data.actions)
                random_q_values = self.critic(replay_data.observations, random_actions)
                random_q_val_loss = sum([th.abs(th.mean(rand_q_val)) for rand_q_val in random_q_values])
                critic_loss = critic_loss + th.mean(random_q_val_loss)

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Approximate the kl-divergence by training the donsker network
            for i in range(4):
                donsker_loss, statistic = get_donsker_loss(
                    self.donsker,
                    replay_data,
                    self.actor,
                    self.behavior_policy,
                    self.gamma,
                )

                self.donsker_optim.zero_grad()
                donsker_loss.backward()
                self.donsker_optim.step()

                donsker_net_losses.append(donsker_loss.item())

                t_var_prod, b_var_prod, rnd = statistic
                max_t_var, _ = th.max(t_var_prod, dim=0)
                max_b_var, _ = th.max(b_var_prod, dim=0)
                max_rnd, _ = th.max(rnd, dim=0)

                mean_t_vars.append(th.mean(t_var_prod).item())
                mean_b_vars.append(th.mean(b_var_prod).item())
                mean_rnds.append(th.mean(rnd).item())

                max_t_vars.append(max_t_var.item())
                max_b_vars.append(max_b_var.item())
                max_rnds.append(max_rnd.item())

            # Compute actor loss
            if self.offline_round_step < WARMUP:
                # During warmup, do behavior cloning
                # mle_loss = -th.mean(self.actor.get_log_prob(replay_data.observations, replay_data.actions))
                mse_loss = F.mse_loss(self.actor(replay_data.observations), replay_data.actions)

                # actor_loss = th.mean(mle_loss) + th.mean(mse_loss)
                actor_loss = mse_loss
                target_bc_losses.append(actor_loss.item())
                if G_USE_PRIOR:
                    b_mean, b_log_std, _ = self.actor.get_action_dist_params(replay_data.observations)
                    b_var = th.exp(b_log_std) ** 2
                    prod_b_var = th.prod(b_var, dim=1)

                    # KL-divergence between the two Gaussians
                    # Since we have prior knowledge on the variance, so regulate only for the variance.
                    var_loss = th.log((G_PRIOR_VARIANCE ** self.action_dim) / prod_b_var) \
                               - self.action_dim + th.sum(b_var / (G_PRIOR_VARIANCE ** self.action_dim), dim=1)

                    actor_loss = actor_loss + th.mean(var_loss)

            else:
                # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
                # Mean over all critic networks
                q_values_pi = th.cat(self.critic.forward(replay_data.observations, actions_pi), dim=1)
                min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)

                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                actor_losses.append(actor_loss.item())

                # donsker_loss = log E [e^{v(s) - B(v(s))}] - (1-gamma) E [v(s_0)]
                # If the donsker network, v, is trained enough, then donsker_loss is a KL-divergence
                # That is, donsker_loss = kl_divergence
                # Here, approx_kl_div is "NEGATIVE" approximated kl divergence
                approx_kl_div = get_approx_kl(
                    self.donsker,
                    replay_data,
                    self.actor,
                    self.behavior_policy,
                    self.gamma,
                )
                actor_loss = 0.005 * actor_loss + approx_kl_div
                approx_kl_means.append(approx_kl_div.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            # th.nn.utils.clip_grad_norm(self.actor.parameters(), 0.5)
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

        # ODICE
        if len(behavior_bc_losses) > 0:
            self.logger.record("train/behavior_bc", np.mean(behavior_bc_losses))
        if len(target_bc_losses) > 0:
            self.logger.record("train/target_bc", np.mean(target_bc_losses))
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record("train/approx_kl_div", np.mean(approx_kl_means))
        self.logger.record("train/donsker_loss", np.mean(donsker_net_losses))
        self.logger.record("statistics/max_t_var", np.mean(max_t_vars))
        self.logger.record("statistics/mean_t_var", np.mean(mean_t_vars))
        self.logger.record("statistics/max_b_var", np.mean(max_b_vars))
        self.logger.record("statistics/mean_b_var", np.mean(mean_b_vars))
        self.logger.record("statistics/max_rnd", np.mean(max_rnds))
        self.logger.record("statistics/mean_rnd", np.mean(mean_rnds))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "SACOdice",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(SACOdice, self).learn(
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
        return super(SACOdice, self)._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
