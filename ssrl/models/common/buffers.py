from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, NamedTuple, Tuple

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


G_FUTURE_THRESH = 3
G_NUM_BUFFER_REPEAT = 256


class History(NamedTuple):
    observations: th.Tensor         # [batch_size, len_subtraj, obs_dim]
    actions: th.Tensor              # [batch_size, len_subtraj, action_dim]


Future = History


class SubtrajBufferSample(NamedTuple):
    observations: th.Tensor     # [batch_size, obs_dim]
    actions: th.Tensor          # [batch_size, action_dim]
    history: History
    future: Future


class GoalcondBufferSample(NamedTuple):
    observations: th.Tensor     # [batch_size, obs_dim]
    actions: th.Tensor          # [batch_size, action_dim]
    history: History
    goal: th.Tensor             # [batch_size, obs_dim]     # goal is one of observatio2n


class ReplayBufferSample(NamedTuple):
    observations: th.Tensor     # [batch_size, obs_dim]
    actions: th.Tensor          # [batch_size, action_dim]


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
    ):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.observation_dim = self.obs_shape[0]
        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = device

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int):
        """
        :param batch_size: Number of element to sample
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray
    ):
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return th.tensor(array, dtype=th.float).to(self.device).detach()
        return th.as_tensor(array).to(self.device).detach()

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        env: Optional[VecNormalize] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


class TrajectoryBuffer(BaseBuffer):
    def __init__(
        self,
        expert_data_path: str,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        max_traj_len: int = 1000,
        device: Union[th.device, str] = "cpu",
    ):
        # Load the expert dataset and set the buffer size by the size of expert dataset
        import pickle
        with open(expert_data_path, "rb") as f:
            expert_dataset = pickle.load(f)        # Dictionary
        buffer_size = len(expert_dataset["observation_trajectories"])

        max_traj_len = max([len(traj) for traj in expert_dataset["observation_trajectories"]])
        super(TrajectoryBuffer, self).__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device
        )
        self.max_traj_len = max_traj_len

        self.expert_dataset = expert_dataset
        self.normalizing = None

        self.observation_traj = np.zeros(
            (self.buffer_size, self.max_traj_len, self.observation_dim), dtype=observation_space.dtype
        )
        self.action_traj = np.zeros(
            (self.buffer_size, self.max_traj_len, self.action_dim), dtype=action_space.dtype
        )
        self.traj_lengths = np.zeros(
            (self.buffer_size, 1), dtype=action_space.dtype
        )

        self.reset()

    @staticmethod
    def timestep_marking(
        history: th.Tensor,
        future: th.Tensor = None,
        device: str = "cuda"
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        History: [batch_size, len_subtraj, obs_dim + action_dim]
        Future: [batch_size, len_subtraj, obs_dim + action_dim]
        Future may be none, especially when evaluation.

        NOTE: History, Future 표현 방식 바꾸려면 여기 바꿔야 함
        Here, we add additional information that the trajectory is whether "history" or "future"

        For history --> -1, -2, -3, ...
        For future --> +1, +2, +3, ...
        """
        batch_size, len_subtraj, _ = history.size()
        history_marker = th.arange(-len_subtraj, 0, device=device).unsqueeze(0) / len_subtraj
        history_marker = history_marker.repeat(batch_size, 1).unsqueeze(-1)
        _history = th.cat((history, history_marker), dim=2)

        if future is not None:
            future_marker = history_marker * -1
            try:
                _future = th.cat((future, future_marker), dim=2)
            except RuntimeError:
                print("len subtraj", len_subtraj)
                print("future size", future.size())
                print("future marker size", future_marker.size())
            return _history, _future

        else:
            return _history, None

    def size(self) -> int:
        pass

    def add(self, *args, **kwargs) -> None:
        pass

    def reset(self) -> None:
        self.traj_lengths[:, 0] = self.expert_dataset["traj_lengths"]
        observations = self.expert_dataset["observation_trajectories"]
        actions = self.expert_dataset["action_trajectories"]
        lengths = self.expert_dataset["traj_lengths"]

        # Add expert data to the trajectory buffer
        for trajectory in range(self.buffer_size):
            traj_length = lengths[trajectory]
            obs_stack = np.vstack(observations[trajectory])     # [traj_length, obs_dim]
            act_stack = np.vstack(actions[trajectory])          # [traj_length, action_dim]
            self.observation_traj[trajectory, :traj_length - 1, :] = obs_stack
            self.action_traj[trajectory, :traj_length - 1, :] = act_stack

        max_obs = np.max(self.observation_traj)
        min_obs = np.min(self.observation_traj)
        normalizing = np.max([max_obs, -min_obs])
        self.normalizing = normalizing
        self.observation_traj /= normalizing

        self.full = True
        self.pos = self.buffer_size

    # def subtraj_sample(self, batch_size: int, len_subtraj: int) -> SubtrajBufferSample:
    #     """
    #     Sample the subtrajectory of the expert data
    #
    #     Note: Batch size is "Maximum" batch size
    #     This is due to that we only collect the subtrajectories
    #     whose front-rear trajectory indices are inside.
    #
    #     즉, 앞뒤로 len_subtraj의 길이를 가지는 subtrajectory를 뽑았을 때, 그 index가
    #     전체 trajectory의 길이를 벗어나지 않는 경우에 대해서만 collect한다
    #     """
    #     low_thresh = 1
    #     high_thresh = self.max_traj_len - len_subtraj
    #     timestep = np.random.randint(low=low_thresh, high=high_thresh - 1)
    #
    #     if timestep < len_subtraj:
    #         len_subtraj = timestep
    #
    #     # 앞뒤로 len_subtraj 만큼 잘랐을 때, index가 넘어가지 않는 친구들
    #     valid_indices, _ = np.nonzero(self.traj_lengths > (timestep + len_subtraj))
    #     assert len(valid_indices) > 0
    #
    #     batch_indices = valid_indices[:batch_size]
    #     current_data = (
    #         self.observation_traj[batch_indices, timestep, :],
    #         self.action_traj[batch_indices, timestep, :]
    #     )
    #     current = tuple(map(self.to_torch, current_data))
    #
    #     history_data = (
    #         self.observation_traj[batch_indices, timestep-len_subtraj:timestep, :],
    #         self.action_traj[batch_indices, timestep-len_subtraj:timestep]
    #     )
    #     history = History(*tuple(map(self.to_torch, history_data)))
    #
    #     # 현재 것 빼고 해야하므로 +1이 붙는 것
    #     future_data = (
    #         self.observation_traj[batch_indices, timestep+1 : timestep+1+len_subtraj, :],
    #         self.action_traj[batch_indices, timestep+1 : timestep+1+len_subtraj, :]
    #     )
    #     future = History(*tuple(map(self.to_torch, future_data)))
    #
    #     return SubtrajBufferSample(*current, history, future)

    def subtraj_sample(
        self,
        batch_size: int,
        history_len_subtraj: int,
        future_len_subtraj: int = None,
        include_current: bool = False
    ) -> SubtrajBufferSample:

        if future_len_subtraj is None:
            future_len_subtraj = history_len_subtraj
        high_thresh = self.max_traj_len - history_len_subtraj

        # 미래 G_FUTURE_THRESH 까지의 state 중 하나를 뽑아서 goal로 설정한다. 또는 초기 state를 뽑아준다
        epsilon = np.random.uniform(0, 1)

        if epsilon < 0.5:       # 앞쪽 부분 학습 (Zero padding 하기 싫다 ~!!!!!!!!!!)
            timestep = np.random.randint(low=1, high=history_len_subtraj + 1)

            valid_indices, *_ = np.nonzero(self.traj_lengths > timestep + future_len_subtraj + 1)

            batch_indices = valid_indices[:batch_size]
            assert len(batch_indices) > 0

            current_data = (
                self.observation_traj[batch_indices, timestep, :],
                self.action_traj[batch_indices, timestep, :]
            )
            current = tuple(map(self.to_torch, current_data))

            upper_bound = timestep + 1 if include_current else timestep
            history_data = (
                self.observation_traj[batch_indices, :upper_bound, :],
                self.action_traj[batch_indices, :upper_bound]
            )
            history = History(*tuple(map(self.to_torch, history_data)))

            future_data = (
                self.observation_traj[batch_indices, timestep+1 : timestep+future_len_subtraj],
                self.action_traj[batch_indices, timestep+1 : timestep+future_len_subtraj]
            )
            future = Future(*(tuple(map(self.to_torch, future_data))))

            return SubtrajBufferSample(*current, history, future)

        else:
            low_thresh = history_len_subtraj
            # future_len = G_FUTURE_THRESH

            # Make a batch mold
            history_len = history_len_subtraj + 1 if include_current else history_len_subtraj
            batch_current_observation = np.zeros((G_NUM_BUFFER_REPEAT, self.observation_dim))
            batch_current_action = np.zeros((G_NUM_BUFFER_REPEAT, self.action_dim))
            batch_history_observation = np.zeros((G_NUM_BUFFER_REPEAT, history_len, self.observation_dim))
            batch_history_action = np.zeros((G_NUM_BUFFER_REPEAT, history_len, self.action_dim))
            batch_future_observation = np.zeros((G_NUM_BUFFER_REPEAT, future_len_subtraj, self.observation_dim))
            batch_future_action = np.zeros((G_NUM_BUFFER_REPEAT, future_len_subtraj, self.action_dim))

            for batch_idx in range(G_NUM_BUFFER_REPEAT):
                timestep = np.random.randint(low=low_thresh, high=high_thresh - future_len_subtraj - 1)
                if timestep < history_len_subtraj:
                    raise NotImplementedError

                valid_indices, *_ = np.nonzero(self.traj_lengths > timestep + future_len_subtraj + 1)
                valid_indices = np.random.permutation(valid_indices)

                batch_indices = valid_indices[0]
                # assert len(batch_indices) > 0

                batch_current_observation[batch_idx] = self.observation_traj[batch_indices, timestep, :]
                batch_current_action[batch_idx] = self.action_traj[batch_indices, timestep, :]

                upper_bound = timestep + 1 if include_current else timestep
                batch_history_observation[batch_idx] \
                    = self.observation_traj[batch_indices, timestep - history_len_subtraj:upper_bound, :]
                batch_history_action[batch_idx] \
                    = self.action_traj[batch_indices, timestep - history_len_subtraj:upper_bound]

                batch_future_observation[batch_idx] \
                    = self.observation_traj[batch_indices, timestep+1 : timestep+1+future_len_subtraj]
                batch_future_action[batch_idx] \
                    = self.action_traj[batch_indices, timestep+1 : timestep+1+future_len_subtraj]

            current_data = (batch_current_observation, batch_current_action)
            current = tuple(map(self.to_torch, current_data))

            history_data = (batch_history_observation, batch_history_action)
            history = History(*tuple(map(self.to_torch, history_data)))

            future_data = (batch_future_observation, batch_future_action)
            future = Future(*tuple(map(self.to_torch, future_data)))

            return SubtrajBufferSample(*current, history, future)

    def _get_samples(
        self, batch_inds: np.ndarray
    ):
        raise NotImplementedError

    def batch_sample(self, batch_size: int) -> SubtrajBufferSample:
        pass


class HindsightBuffer(TrajectoryBuffer):
    def __init__(
        self,
        expert_data_path: str,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        max_traj_len: int = 1000,
        device: Union[th.device, str] = "cpu",
    ):
        super(HindsightBuffer, self).__init__(
            expert_data_path=expert_data_path,
            observation_space=observation_space,
            action_space=action_space,
            max_traj_len=max_traj_len,
            device=device,
        )

    def size(self) -> int:
        pass

    def add(self, *args, **kwargs) -> None:
        pass

    def reset(self) -> None:
        super().reset()

    def _get_samples(
        self, batch_inds: np.ndarray
    ):
        pass

    def goalcond_sample(
        self,
        batch_size: int,
        len_subtraj: int,
        include_current: bool = False
    ) -> GoalcondBufferSample:

        high_thresh = self.max_traj_len - len_subtraj

        # 미래 G_FUTURE_THRESH 까지의 state 중 하나를 뽑아서 goal로 설정한다. 또는 초기 state를 뽑아준다
        epsilon = np.random.uniform(0, 1)

        if epsilon < 0.5:       # 앞쪽 부분 학습
            timestep = np.random.randint(low=1, high=len_subtraj + 1)

            valid_indices, *_ = np.nonzero(self.traj_lengths > timestep + G_FUTURE_THRESH + 1)

            if timestep < len_subtraj:
                len_subtraj = timestep

            batch_indices = valid_indices[:batch_size]
            assert len(batch_indices) > 0

            current_data = (
                self.observation_traj[batch_indices, timestep, :],
                self.action_traj[batch_indices, timestep, :]
            )
            current = tuple(map(self.to_torch, current_data))

            upper_bound = timestep + 1 if include_current else timestep
            history_data = (
                self.observation_traj[batch_indices, timestep - len_subtraj:upper_bound, :],
                self.action_traj[batch_indices, timestep - len_subtraj:upper_bound]
            )
            history = History(*tuple(map(self.to_torch, history_data)))

            goal_indices = np.random.randint(
                low=timestep,
                high=timestep + np.ones_like(self.traj_lengths[batch_indices]) * G_FUTURE_THRESH
            ).squeeze()

            goal_data = (self.observation_traj[batch_indices, goal_indices, ...])
            goal = self.to_torch(goal_data)
            return GoalcondBufferSample(*current, history, goal)

        else:
            low_thresh = len_subtraj

            # Make a batch mold
            history_len = len_subtraj + 1 if include_current else len_subtraj
            batch_current_observation = np.zeros((G_NUM_BUFFER_REPEAT, self.observation_dim))
            batch_current_action = np.zeros((G_NUM_BUFFER_REPEAT, self.action_dim))
            batch_goal_observation = np.zeros((G_NUM_BUFFER_REPEAT, self.observation_dim))
            batch_history_observation = np.zeros((G_NUM_BUFFER_REPEAT, history_len, self.observation_dim))
            batch_history_action = np.zeros((G_NUM_BUFFER_REPEAT, history_len, self.action_dim))

            for batch_idx in range(G_NUM_BUFFER_REPEAT):
                timestep = np.random.randint(low=low_thresh, high=high_thresh - G_FUTURE_THRESH - 1)
                if timestep < len_subtraj:
                    raise NotImplementedError

                valid_indices, *_ = np.nonzero(self.traj_lengths > timestep + G_FUTURE_THRESH + 1)
                valid_indices = np.random.permutation(valid_indices)

                batch_indices = valid_indices[0]
                # assert len(batch_indices) > 0

                batch_current_observation[batch_idx] = self.observation_traj[batch_indices, timestep, :]
                batch_current_action[batch_idx] = self.action_traj[batch_indices, timestep, :]

                upper_bound = timestep + 1 if include_current else timestep
                batch_history_observation[batch_idx] \
                    = self.observation_traj[batch_indices, timestep - len_subtraj:upper_bound, :]
                batch_history_action[batch_idx] \
                    = self.action_traj[batch_indices, timestep - len_subtraj:upper_bound]

                goal_indices = np.random.randint(
                    low=timestep,
                    high=timestep + np.ones_like(self.traj_lengths[batch_indices]) * G_FUTURE_THRESH
                ).squeeze()
                batch_goal_observation[batch_idx] \
                    = self.observation_traj[batch_indices, goal_indices]

            current_data = (batch_current_observation, batch_current_action)
            current = tuple(map(self.to_torch, current_data))

            history_data = (batch_history_observation, batch_history_action)
            history = History(*tuple(map(self.to_torch, history_data)))

            goal_data = batch_goal_observation
            goal = self.to_torch(goal_data)

            return GoalcondBufferSample(*current, history, goal)


class ReplayBuffer(BaseBuffer):
    def __init__(
        self,
        expert_data_path: str,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
    ):
        # Load the expert dataset and set the buffer size by the size of expert dataset
        import pickle
        with open(expert_data_path, "rb") as f:
            expert_dataset = pickle.load(f)  # Dictionary

        n_trajectory = len(expert_dataset["observation_trajectories"])
        buffer_size = sum([len(traj) for traj in expert_dataset["observation_trajectories"]])

        super(ReplayBuffer, self).__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device
        )

        self.expert_dataset = expert_dataset
        self.observations = None
        self.actions = None
        self.n_trajectory = n_trajectory

        self.reset()

    def add(self, *args, **kwargs) -> None:
        pass

    def extend(self, *args, **kwargs) -> None:
        pass

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.observation_dim), dtype=self.observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=self.action_space.dtype)

        top_pos = 0
        for traj in range(self.n_trajectory):
            ith_observations = self.expert_dataset["observation_trajectories"][traj]
            ith_actions = self.expert_dataset["action_trajectories"][traj]
            traj_length = len(ith_observations)

            self.observations[top_pos: top_pos + traj_length, ...] = np.vstack(ith_observations)
            self.actions[top_pos: top_pos + traj_length, ...] = np.vstack(ith_actions)
            top_pos += traj_length

        self.pos = 0
        self.full = True

    def batch_sample(self, batch_size: int):
        batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size

        data = (
            self.observations[batch_inds, :],
            self.actions[batch_inds, :],
        )
        return ReplayBufferSample(*tuple(map(self.to_torch, data)))

    def _get_samples(
        self, batch_inds: np.ndarray
    ):
        pass


if __name__ == "__main__":
    import gym
    env = gym.make("MountainCarContinuous-v0")
    buffer = HindsightBuffer(
        expert_data_path="/workspace/expertdata/MountainCarContinuous-v0/expert_buffer-10",
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
    z = buffer.goalcond_sample(batch_size=10, len_subtraj=100)
    for i in range(10000):
        z = buffer.goalcond_sample(
            batch_size=10,
            len_subtraj=100
        )
        print(i)
