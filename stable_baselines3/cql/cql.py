from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F
from collections import deque

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.cql.policies import CQLPolicy

KUMER_STYLE = True
# CONSERVATIVE_WEIGHT = 10.0


class CQL(OffPolicyAlgorithm):
    """
    Conservative Q Learning (CQL)
    We implement CQL(H) as in equation(4) and appendix F.

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param alpha_coef: Initial alpha, a coefficient of conservative loss : See equation (4) of the paper.
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param num_randoms: A number of importance sampling : How much sampling to calculate the equation (4) in the paper.
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
        policy: Union[str, Type[CQLPolicy]],
        env: Union[GymEnv, str],
        dataset: Tuple = None,
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

        # Add for CQL
        alpha_coef: float = "auto",
        num_randoms: int = 10,
        lagrange_thresh: int = 10.0,
        without_exploration: bool = True,
        conservative_weight: float = 10.0,
        gumbel_ensemble: bool = False,
        gumbel_temperature: float = 0.5,
    ):
        super(CQL, self).__init__(
            policy,
            env,
            CQLPolicy,
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
            supported_action_spaces=(gym.spaces.Box, gym.spaces.Discrete,),

            without_exploration=without_exploration,
            gumbel_ensemble=gumbel_ensemble,
            gumbel_temperature=gumbel_temperature
        )

        self.num_randoms = num_randoms
        self.lagrange_thresh = lagrange_thresh
        self.dataset = dataset

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer = None

        self.alpha_coef = alpha_coef
        self.alpha_coef_optimizer = None

        self.conservative_weight = conservative_weight

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(CQL, self)._setup_model()
        self._create_aliases()
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The alpha coefficient of CQL can be learned automatically
        if isinstance(self.alpha_coef, str) and self.alpha_coef.startswith("auto"):
            init_alpha = 0.1
            self.log_alpha_coef = th.log(th.ones(1, device=self.device) * init_alpha).requires_grad_(True)
            self.alpha_coef_optimizer = th.optim.Adam([self.log_alpha_coef], lr=self.lr_schedule(1))
            # self.alpha_coef_optimizer = th.optim.Adam([self.log_alpha_coef], lr=1e-3)

        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.alpha_coef_tensor = th.tensor(float(self.alpha_coef)).to(self.device)

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

        # Deal with offline reinforcement learning, as original object of CQL.

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
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
        # [batch_size * num_randoms, observation_dim]
        observations = th.repeat_interleave(replay_data.observations, repeats=self.num_randoms, dim=0)
        with th.no_grad():
            current_actions, current_log_probs = self.actor.action_log_prob(observations)
        # current_actions: [batch_size, num_randoms, num_actions]

        current_actions = current_actions.reshape(batch_size, self.num_randoms, -1)

        # current_log_probs: [batch_size, num_randoms]
        current_log_probs = current_log_probs.reshape(batch_size, self.num_randoms, )

        random_actions \
            = th.rand((batch_size, self.num_randoms, num_actions), requires_grad=True).to(self.device) * 2 - 1

        random_action_q_values = list(self.critic.cql_forward(observations, random_actions))
        current_action_q_values = list(self.critic.cql_forward(observations, current_actions))
        num_critics = len(random_action_q_values)
        random_density = th.log(th.tensor(0.5 ** num_actions))

        cat_input = [[qj_rand_q_val - random_density, qj_curr_q_val - current_log_probs]
                     for qj_rand_q_val, qj_curr_q_val in zip(random_action_q_values, current_action_q_values)]

        q_cats = [th.cat(x, dim=1) for x in cat_input]      # Length = n_critics
        
        min_q_losses = [th.logsumexp(q_cats[j], dim=1).mean() - current_q_values[j].mean() for j in range(num_critics)]
        conservative_loss = sum(min_q_losses) / num_critics

        if self.conservative_weight > 0:
            return self.conservative_weight * conservative_loss
        else:
            return conservative_loss

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # For CQL dual gradient
        alpha_coefs = []
        if self.alpha_coef_optimizer is not None:
            alpha_coef_losses = []

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        max_qvals, min_qvals, mean_qvals = [], [], []

        conservative_losses = []
        for gradient_step in range(1):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size)
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

                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            """
                S T A R T:
                  Added for CQL(H)
            """
            conservative_loss = self._get_conservative_loss(replay_data, current_q_values)
            # We can update alpha by using dual gradient method.
            alpha_coef_loss = None
            if self.alpha_coef_optimizer is not None:
                alpha_coef = th.exp(self.log_alpha_coef.detach())
                # no grad 안하면 gradient 섞임.
                alpha_coef_loss = -th.exp(self.log_alpha_coef) * (conservative_loss.detach() - self.lagrange_thresh)
                alpha_coef_losses.append(alpha_coef_loss.item())
            else:
                alpha_coef = self.alpha_coef_tensor

            # For stability, we clamp to 0 ~ 1e6
            alpha_coef = alpha_coef.clamp(0, 1e6)
            alpha_coefs.append(alpha_coef.item())

            if alpha_coef_loss is not None:
                self.alpha_coef_optimizer.zero_grad()
                alpha_coef_loss.backward()
                self.alpha_coef_optimizer.step()

            # Compute critic loss
            original_critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])

            # Compute conservative loss
            conservative_losses.append(conservative_loss.item())
            conservative_loss = alpha_coef * (conservative_loss - self.lagrange_thresh)
            critic_loss = conservative_loss + original_critic_loss

            critic_losses.append(original_critic_loss.item())
            """
                E N D:
                    Added for CQL(H)
            """

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = th.cat(self.critic.forward(replay_data.observations, actions_pi), dim=1)

            max_qf_pi, _ = th.max(q_values_pi, dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            # Add for logging

            max_qvals.append(max_qf_pi.mean().item())
            min_qvals.append(min_qf_pi.mean().item())
            mean_qvals.append(q_values_pi.mean().item())

            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
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
        self.logger.record("train/alpha_coef", np.mean(alpha_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

        # CQL
        self.logger.record("config/conservative_weight", self.conservative_weight)
        self.logger.record("config/lag_thresh", self.lagrange_thresh)
        self.logger.record("train/conservative_loss", np.mean(conservative_losses))
        self.logger.record("train/alpha_coef_loss", np.mean(alpha_coef_losses))
        self.logger.record("train/max_qvals", np.mean(max_qvals))
        self.logger.record("train/mean_qvals", np.mean(mean_qvals))
        self.logger.record("train/min_qvals", np.mean(min_qvals))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "CQL",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(CQL, self).learn(
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
        return super(CQL, self)._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
