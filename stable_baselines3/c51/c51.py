from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, is_vectorized_observation, polyak_update
from stable_baselines3.c51.policies import C51Policy


class C51(OffPolicyAlgorithm):
    """
        C51 Network

        :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
        :param env: The environment to learn from (if registered in Gym, can be str)
        :param learning_rate: The learning rate, it can be a function
            of the current progress remaining (from 1 to 0)
        :param buffer_size: size of the replay buffer
        :param learning_starts: how many steps of the model to collect transitions for before learning starts
        :param batch_size: Minibatch size for each gradient update
        :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
        :param gamma: the discount factor
        :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
            like ``(5, "step")`` or ``(2, "episode")``.
        :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
            Set to ``-1`` means to do as many gradient steps as steps done in the environment
            during the rollout.
        :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
            If ``None``, it will be automatically selected.
        :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
        :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
            at a cost of more complexity.
            See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        :param target_update_interval: update the target network every ``target_update_interval``
            environment steps.
        :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
        :param exploration_initial_eps: initial value of random action probability
        :param exploration_final_eps: final value of random action probability
        :param max_grad_norm: The maximum value for the gradient clipping
        :param tensorboard_log: the log location for tensorboard (if None, no logging)
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
        policy: Union[str, Type[C51Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1000000,  # 1e6
        learning_starts: int = 50000,
        batch_size: Optional[int] = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        num_atoms: int = 51,
        min_rewards: int = 0,
        max_rewards: int = 10,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(C51, self).__init__(
            policy,
            env,
            C51Policy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
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
            supported_action_spaces=(gym.spaces.Discrete,),
        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        self.num_atoms = num_atoms
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
        self.distributional_q_net, self.distributional_q_net_target = None, None
        self.min_rewards, self.max_rewards = min_rewards, max_rewards
        self.atoms = th.linspace(min_rewards, max_rewards, steps=num_atoms, device=self.device)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(C51, self)._setup_model()
        self._create_aliases()
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

    def _create_aliases(self) -> None:
        self.distributional_q_net = self.policy.dist_q_net
        self.distributional_q_net_target = self.policy.dist_q_net_target

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        if self.num_timesteps % self.target_update_interval == 0:
            polyak_update(
                self.distributional_q_net.parameters(), self.distributional_q_net_target.parameters(), self.tau
            )

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration rate", self.exploration_rate)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the distribution of returns according to the distributional Q network.

                next_q_values_distributions =\
                    self.distributional_q_net_target(replay_data.next_observations)   # [batch_size, action_dim, n_atom]

                # Prediction of next action is by using a q network, not a target network.
                next_actions = self.distributional_q_net_target._predict(replay_data.next_observations)   # [batch_size]

                # print(replay_data.rewards)
                target_probs = self._projection_and_get_target(
                    replay_data.rewards, next_actions, next_q_values_distributions
                )

            current_q_values_distributions = \
                self.distributional_q_net(replay_data.observations)      # [batch_size, action_dim, n_atom]

            predicted_probs = current_q_values_distributions[th.arange(batch_size), replay_data.actions.squeeze()]

            # loss = th.sum(target_probs * th.log(predicted_probs)) / batch_size
            loss = F.kl_div(th.log(predicted_probs), target_probs)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def _projection_and_get_target(self, rewards, next_actions, next_q_values_distributions):
        """
        :param rewards: [batch_size, 1]
        :param next_actions: [batch_size], obtained from behavior network.
        :param next_q_values_distributions: [batch_size, action_dim, n_atoms], obtained from target network.

        :return: A projected distribution, which is an m_i of the algorithm in the paper [batch_size, n_atoms]
        """
        batch_size, action_dim, n_atoms = next_q_values_distributions.size()

        device = next_q_values_distributions.device
        target_probs = th.zeros((batch_size, n_atoms)).to(device)

        atoms_target = th.clip(rewards + self.gamma * self.atoms, self.min_rewards, self.max_rewards)

        upper_idx = th.ceil(atoms_target)
        lower_idx = upper_idx - 1

        # TODO : 여기 For문 바꾸기
        for i in range(batch_size):
            next_action = next_actions[i].item()
            for j in range(n_atoms):
                l_idx = int(lower_idx[i, j].item())
                u_idx = int(upper_idx[i, j].item())
                target_probs[i, l_idx] += next_q_values_distributions[i, next_action, j] * (u_idx - atoms_target[i, j])
                target_probs[i, u_idx] += next_q_values_distributions[i, next_action, j] * (atoms_target[i, j] - l_idx)

        return target_probs

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if is_vectorized_observation(maybe_transpose(observation, self.observation_space),
                                         self.observation_space):
                if isinstance(self.observation_space, gym.spaces.Dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, mask, deterministic)
        return action, state

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "DQN",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(C51, self).learn(
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
        return super(C51, self)._excluded_save_params() + ["distributional_q_net", "distributional_q_net_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
