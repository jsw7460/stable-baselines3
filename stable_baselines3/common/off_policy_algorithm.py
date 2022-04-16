import io
import pathlib
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import gzip
from concurrent import futures
import d4rl
from collections import deque

import gym
import numpy as np
import torch as th
import torch.nn.functional as F

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

STORE_FILENAME_PREFIX = '$store$_'
DEBUG = True


class OffPolicyAlgorithm(BaseAlgorithm):
    """
    The base for Off-Policy algorithms (ex: SAC/TD3)

    :param policy: Policy object
    :param env: The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param policy_base: The base policy used by this method
    :param learning_rate: learning rate for the optimizer,
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
    :param policy_kwargs: Additional arguments to be passed to the policy on creation
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug
    :param device: Device on which the code should run.
        By default, it will try to use a Cuda compatible device and fallback to cpu
        if it is not possible.
    :param support_multi_env: Whether the algorithm supports training
        with multiple environments (as in A2C)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param seed: Seed for the pseudo random generators
    :param use_sde: Whether to use State Dependent Exploration (SDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param sde_support: Whether the model support gSDE or not
    :param remove_time_limit_termination: Remove terminations (dones) that are due to time limit.
        See https://github.com/hill-a/stable-baselines/issues/863
    :param supported_action_spaces: The action spaces supported by the algorithm.
    :param without_exploration: True only when offline reinforcement learning.
    """

    def __init__(
        self,
        policy: Type[BasePolicy],
        env: Union[GymEnv, str],
        policy_base: Type[BasePolicy],
        learning_rate: Union[float, Schedule],
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "step"),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        support_multi_env: bool = False,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        sde_support: bool = True,
        remove_time_limit_termination: bool = False,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
        without_exploration: bool = False,
        gumbel_ensemble: bool = False,
        gumbel_temperature: float = 0.5,
        dropout: float = 0.0,
    ):
        self.d4rl_env = None
        self.without_exploration = without_exploration
        if without_exploration:
            self.d4rl_env = env
        super(OffPolicyAlgorithm, self).__init__(
            policy=policy,
            env=env,
            policy_base=policy_base,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=support_multi_env,
            create_eval_env=create_eval_env,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            supported_action_spaces=supported_action_spaces,
        )

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.tau = tau
        self.gamma = gamma
        self.gradient_steps = gradient_steps
        self.action_noise = action_noise
        self.optimize_memory_usage = optimize_memory_usage
        self.replay_buffer_class = replay_buffer_class
        if replay_buffer_kwargs is None:
            replay_buffer_kwargs = {}
        self.replay_buffer_kwargs = replay_buffer_kwargs
        self._episode_storage = None

        # Remove terminations (dones) that are due to time limit
        # see https://github.com/hill-a/stable-baselines/issues/863
        self.remove_time_limit_termination = remove_time_limit_termination

        # Save train freq parameter, will be converted later to TrainFreq object
        self.train_freq = train_freq

        self.actor = None  # type: Optional[th.nn.Module]
        self.replay_buffer = None  # type: Optional[ReplayBuffer]
        # Update policy keyword arguments
        if sde_support:
            self.policy_kwargs["use_sde"] = self.use_sde
        # For gSDE only
        self.use_sde_at_warmup = use_sde_at_warmup

        self.gumbel_ensemble = gumbel_ensemble
        self.gumbel_temperature = gumbel_temperature
        # Added for offline reinforcement learning.

        self.offline_rewards = deque(maxlen=10)
        self.offline_normalized_rewards = deque(maxlen=10)
        self.offline_rewards_std = deque(maxlen=10)

        self.dropout = dropout

        if without_exploration:
            self.reload_buffer = True
            self.offline_round_step = 0
            self.without_exploration = True
            self.buffer_load_interval = None
            self.offline_mean_reward = 0
            self.offline_evaluation_step = 0
            self.data_dir = None
            self.goal_data = None
            self.action_dim = self.action_space.shape[0]
            if env.__str__() == "<AntMazeEnv<AntULongTestEnv-v0>>":
                self.state_dim = 31
            else:
                self.state_dim = self.observation_space.shape[0]

    def get_gumbel_coefs(self, q_values: th.Tensor, inverse_proportion: bool = False) -> th.Tensor:
        """
        q_values: [batch_size, n_critics].
        Return the gumbel softmax value of the critics.
        """
        if inverse_proportion:      # "Minimum" would have a higher value "in average".
            q_values = -q_values
        coefs = F.gumbel_softmax(q_values, tau=self.gumbel_temperature, hard=False, dim=1)
        return coefs

    def _load_d4rl_env(self) -> None:
        dataset = d4rl.qlearning_dataset(self.d4rl_env)
        observations = dataset["observations"]
        next_observations = dataset["next_observations"]
        actions = dataset["actions"]
        rewards = dataset["rewards"]
        dones = dataset["terminals"]
        buffer_size = len(observations)
        infos = [None for _ in range(buffer_size)]      # Episode당 information정보만 들어있음. 학습에 영향 X.

        for obs, next_obs, action, reward, done, info \
                in zip(observations, next_observations, actions, rewards, dones, infos):
            # 이거 if문 안하면 base_class 파일의 ~~or self._last_obs is None: 줄로 들어가져서, env랑 소통하게 된다.
            if self._last_obs is None:
                self._last_obs = obs
            self._store_transition(
                self.replay_buffer,
                action,
                next_obs,
                reward,
                done,
                info
            )
            if self.replay_buffer.full:
                break

    def _convert_train_freq(self) -> None:
        """
        Convert `train_freq` parameter (int or tuple)
        to a TrainFreq object.
        """
        if not isinstance(self.train_freq, TrainFreq):
            train_freq = self.train_freq

            # The value of the train frequency will be checked later
            if not isinstance(train_freq, tuple):
                train_freq = (train_freq, "step")

            try:
                train_freq = (train_freq[0], TrainFrequencyUnit(train_freq[1]))
            except ValueError:
                raise ValueError(f"The unit of the `train_freq` must be either 'step' or 'episode' not '{train_freq[1]}'!")

            if not isinstance(train_freq[0], int):
                raise ValueError(f"The frequency of `train_freq` must be an integer and not {train_freq[0]}")

            self.train_freq = TrainFreq(*train_freq)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # Use DictReplayBuffer if needed
        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, gym.spaces.Dict):
                self.replay_buffer_class = DictReplayBuffer
            else:
                self.replay_buffer_class = ReplayBuffer

        elif self.replay_buffer_class == HerReplayBuffer:
            assert self.env is not None, "You must pass an environment when using `HerReplayBuffer`"

            # If using offline sampling, we need a classic replay buffer too
            if self.replay_buffer_kwargs.get("online_sampling", True):
                replay_buffer = None
            else:
                replay_buffer = DictReplayBuffer(
                    self.buffer_size,
                    self.observation_space,
                    self.action_space,
                    self.device,
                    optimize_memory_usage=self.optimize_memory_usage,
                )

            self.replay_buffer = HerReplayBuffer(
                self.env,
                self.buffer_size,
                self.device,
                replay_buffer=replay_buffer,
                **self.replay_buffer_kwargs,
            )

        if self.replay_buffer is None:
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                self.device,
                optimize_memory_usage=self.optimize_memory_usage,
                **self.replay_buffer_kwargs,
            )

        self.policy_kwargs["dropout"] = self.dropout
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )

        self.policy = self.policy.to(self.device)

        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()

    def save_replay_buffer(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        """
        Save the replay buffer as a pickle file.

        :param path: Path to the file where the replay buffer should be saved.
            if path is a str or pathlib.Path, the path is automatically created if necessary.
        """
        assert self.replay_buffer is not None, "The replay buffer is not defined"
        save_to_pkl(path, self.replay_buffer, self.verbose)

    def load_replay_buffer(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        truncate_last_traj: bool = True,
    ) -> None:
        """
        Load a replay buffer from a pickle file.

        :param path: Path to the pickled replay buffer.
        :param truncate_last_traj: When using ``HerReplayBuffer`` with online sampling:
            If set to ``True``, we assume that the last trajectory in the replay buffer was finished
            (and truncate it).
            If set to ``False``, we assume that we continue the same trajectory (same episode).
        """
        self.replay_buffer = load_from_pkl(path, self.verbose)
        assert isinstance(self.replay_buffer, ReplayBuffer), "The replay buffer must inherit from ReplayBuffer class"

        # Backward compatibility with SB3 < 2.1.0 replay buffer
        # Keep old behavior: do not handle timeout termination separately
        if not hasattr(self.replay_buffer, "handle_timeout_termination"):  # pragma: no cover
            self.replay_buffer.handle_timeout_termination = False
            self.replay_buffer.timeouts = np.zeros_like(self.replay_buffer.dones)

        if isinstance(self.replay_buffer, HerReplayBuffer):
            assert self.env is not None, "You must pass an environment at load time when using `HerReplayBuffer`"
            self.replay_buffer.set_env(self.get_env())
            if truncate_last_traj:
                self.replay_buffer.truncate_last_trajectory()

    def load_trunc_replay_buffer(self, path, load_size: int = 10000):
        np.random.seed(self.seed)
        full_buffer = load_from_pkl(path, self.verbose)
        idx = np.random.permutation(load_size)
        self.replay_buffer.observations = full_buffer.observations[idx, ...]
        self.replay_buffer.actions = full_buffer.actions[idx, ...]
        self.replay_buffer.rewards = full_buffer.rewards[idx, ...]
        self.replay_buffer.next_observations = full_buffer.next_observations[idx, ...]
        self.replay_buffer.full = True
        self.reload_buffer = False

    def register_rewards(self, reward_mean, reward_std=None, reduction=True):
        self.offline_mean_reward += reward_mean
        if reduction:
            self.offline_mean_reward = self.offline_mean_reward / 2
            self.offline_evaluation_step += 1

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        callback: MaybeCallback = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        without_exploration: bool = False,
    ) -> Tuple[int, BaseCallback]:
        """
        cf `BaseAlgorithm`.
        """
        # Prevent continuity issue by truncating trajectory
        # when using memory efficient replay buffer
        # see https://github.com/DLR-RM/stable-baselines3/issues/46

        # Special case when using HerReplayBuffer,
        # the classic replay buffer is inside it when using offline sampling

        if self.without_exploration and self.reload_buffer:        # Add for offline RL.
            # self._load_d4rl_env()
            self.reload_buffer = False

        if isinstance(self.replay_buffer, HerReplayBuffer):
            replay_buffer = self.replay_buffer.replay_buffer
        else:
            replay_buffer = self.replay_buffer

        truncate_last_traj = (
            self.optimize_memory_usage
            and reset_num_timesteps
            and replay_buffer is not None
            and (replay_buffer.full or replay_buffer.pos > 0)
        )

        if truncate_last_traj:
            warnings.warn(
                "The last trajectory in the replay buffer will be truncated, "
                "see https://github.com/DLR-RM/stable-baselines3/issues/46."
                "You should use `reset_num_timesteps=False` or `optimize_memory_usage=False`"
                "to avoid that issue."
            )
            # Go to the previous index
            pos = (replay_buffer.pos - 1) % replay_buffer.buffer_size
            replay_buffer.dones[pos] = True

        return super()._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            log_path,
            reset_num_timesteps,
            tb_log_name,
            without_exploration=self.without_exploration,
        )

    def load_expert_data(self, buffer, path: str, env: gym.Env) -> None:
        self.replay_buffer = buffer(
            expert_data_path=path,
            observation_space=env.observation_space,
            action_space=env.action_space,
            max_traj_len=1000,
            device=self.device
        )

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OffPolicyAlgorithm":
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
            without_exploration=self.without_exploration
        )

        callback.on_training_start(locals(), globals())
        while self.num_timesteps < total_timesteps:
            if not self.without_exploration:             # Original
                rollout = self.collect_rollouts(
                    self.env,
                    train_freq=self.train_freq,
                    action_noise=self.action_noise,
                    callback=callback,
                    learning_starts=self.learning_starts,
                    replay_buffer=self.replay_buffer,
                    log_interval=log_interval,
                )
                if rollout.continue_training is False:
                    break
                if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                    # If no `gradient_steps` is specified,
                    # do as many gradients steps as steps performed during the rollout
                    gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                    # Special case when the user passes `gradient_steps=0`
                    if gradient_steps > 0:
                        self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

            else:    # For offline RL. No collect rollouts.
                gradient_steps = self.gradient_steps
                self.num_timesteps += gradient_steps
                self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        if self.without_exploration:
            self.offline_round_step += 1
        callback.on_training_end()
        return self

    def collect_data_and_save(self, path: str, save_size: int = None) -> None:
        _, callback = self._setup_learn(1, None)
        if save_size is None:
            save_size = self.replay_buffer.buffer_size
        while self.replay_buffer.pos < save_size:
            _ = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=9999999999,
            )
        self.save_replay_buffer(path + f"-size{save_size}")

    def train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Sample the replay buffer and do the updates
        (gradient descent and update target networks)
        """
        raise NotImplementedError()

    def _sample_action(
        self, learning_starts: int, action_noise: Optional[ActionNoise] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample()])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def dump_logs(self) -> None:
        self._dump_logs()

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        time_elapsed = time.time() - self.start_time
        fps = int(self.num_timesteps / (time_elapsed + 1e-8))

        algo_name = self.__class__.__name__.split(".")[-1]
        try:
            env_name = self.env.get_attr("unwrapped", 0)[0].spec.id
        except AttributeError:
            env_name = None

        self.logger.record("config/Algorithm", algo_name, exclude="tensorboard")
        if env_name is not None:
            self.logger.record("config/Environment", env_name, exclude="tensorboard")
        try:
            self.logger.record("config/Env-state", self.observation_space.shape[0], exclude="tensorboard")
            self.logger.record("config/Env-action", self.action_space.shape[0], exclude="tensorboard")
        except TypeError:
            pass

        if self.gumbel_ensemble and self.gumbel_temperature > 0:
            self.logger.record("train/Gumbel_temperature", self.gumbel_temperature, exclude="tensorboard")
        if self.without_exploration:
            self.logger.record("time/offline_rounds", self.offline_round_step + 1, exclude="tensorboard")
            if self.policy_kwargs.get("n_critics") is not None:
                self.logger.record("config/n_critics", self.policy_kwargs["n_critics"], exclude="tensorboard")
            # self.logger.record("train/mean_rewards", self.offline_mean_reward)
        else:
            self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
            self.logger.record("time/fps", fps)
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        if len(self.offline_rewards) > 0:
            self.logger.record("performance/reward/mean", np.mean(self.offline_rewards))
        if len(self.offline_rewards_std) > 0:
            self.logger.record("performance/reward/std", np.mean(self.offline_rewards_std), exclude="tensorboard")
        if len(self.offline_normalized_rewards) > 0:
            self.logger.record("performance/normalized_rewards_mean", np.mean(self.offline_normalized_rewards))
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))

        self.logger.record("config/seed", self.seed, exclude="tensorboard")
        self.logger.record("config/batch_size", self.batch_size, exclude="tensorboard")
        self.logger.record("config/buffer_size", self.replay_buffer.buffer_size, exclude="tensorboard")
        self.logger.record("config/device", self.device, exclude="tensorboard")
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def _on_step(self) -> None:
        """
        Method called after each step in the environment.
        It is meant to trigger DQN target network update
        but can be used for other purposes
        """
        pass

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when done is True)
        :param reward: reward for the current transition
        :param done: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version

        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode

        try:
            if done and infos[0].get("terminal_observation") is not None:
                next_obs = infos[0]["terminal_observation"]
                # VecNormalize normalizes the terminal observation
                if self._vec_normalize_env is not None:
                    next_obs = self._vec_normalize_env.unnormalize_obs(next_obs)
            else:
                next_obs = new_obs_
        except:
            next_obs = new_obs_
            pass
        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            done,
            infos,
        )
        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    @staticmethod
    def collect_expert_traj(
        model,
        env: gym.Env,
        save_data_path: str,
        collect_size: int = 10000,
        deterministic: bool = True,
        perturb: float = 0.0,
        pomdp_hidden_dim: int = 0,
    ) -> None:
        import pickle
        # 아래의 List 안에 서로 다른 길이의 trajectory들을 저장 할 것이다
        observation_trajectories = []
        action_trajectories = []
        reward_trajectories = []
        info_trajectories = []

        episode_rewards = []

        traj_lengths = []

        for single_traj in range(collect_size):
            print(f"{single_traj}th run..")
            observation = env.reset()
            j = 0
            done = False
            episode_reward, episode_timesteps = 0.0, 0

            # 위의 *_trajectories 안에 서로 다른 길이의 trajectory list를 넣어 줄 것이다
            obs_traj, act_traj, rew_traj, info_traj = [], [], [], []
            traj_len = 1
            while not done:
                j += 1
                prob = np.random.uniform(0, 1)
                if prob < perturb:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(observation, state=None, deterministic=deterministic)
                new_obs, reward, done, infos = env.step(action)
                print("new obs", new_obs)
                episode_reward += reward

                traj_len += 1
                obs_traj.append(observation)
                act_traj.append(action)
                rew_traj.append(reward)
                info_traj.append(infos)
                observation = new_obs.copy()

            episode_rewards.append(episode_reward)
            traj_lengths.append(traj_len)

            observation_trajectories.append(obs_traj)
            action_trajectories.append(act_traj)
            reward_trajectories.append(rew_traj)
            info_trajectories.append(info_traj)

        assert len(observation_trajectories) == len(action_trajectories) \
               and len(action_trajectories) == len(reward_trajectories)

        expert_dataset = {
            "observation_trajectories": observation_trajectories,
            "action_trajectories": action_trajectories,
            "reward_trajectories": reward_trajectories,
            "info_trajectories": info_trajectories,
            "traj_lengths": traj_lengths,
        }
        if pomdp_hidden_dim > 0:
            expert_dataset["pomdp_hidden_dim"] = pomdp_hidden_dim

        with open(save_data_path, "wb") as f:
            pickle.dump(expert_dataset, f)

        print("Dataset Statistics")
        print("------------------------------")
        print("\t The Number of Trajectories:", collect_size)
        print("\t Mean Length of Trajectories:", np.mean(traj_lengths))
        print("\t Mean Reward of Episodes:", np.mean(episode_rewards))
        print("------------------------------")

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
        offline_dataset: Dict = None
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.
        :param env: The training environment
        :param callback: Callback that will be called at each stepㄱ
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :param offline_dataset: a dataset for offline reinforcement learning.
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        episode_rewards, total_timesteps = [], []
        num_collected_steps, num_collected_episodes = 0, 0

        assert not self.without_exploration, "Collecting rollout admissible only for online RL"
        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            done = False
            episode_reward, episode_timesteps = 0.0, 0

            while not done:
                if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                action, buffer_action = self._sample_action(learning_starts, action_noise)

                # Rescale and perform action
                new_obs, reward, done, infos = env.step(action)
                self.num_timesteps += 1
                episode_timesteps += 1
                num_collected_steps += 1

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, num_collected_steps, num_collected_episodes, continue_training=False)

                episode_reward += reward

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)

                # Store data in replay buffer (normalized action and unnormalized observation)

                self._store_transition(replay_buffer, buffer_action, new_obs, reward, done, infos)

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()

                if not should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
                    break

            if done:
                num_collected_episodes += 1
                self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

        mean_reward = np.mean(episode_rewards) if num_collected_episodes > 0 else 0.0

        callback.on_rollout_end()
        return RolloutReturn(mean_reward, num_collected_steps, num_collected_episodes, continue_training)
