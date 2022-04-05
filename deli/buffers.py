import collections
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, NamedTuple, Tuple

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class History(NamedTuple):
    observations: th.Tensor         # [batch_size, len_subtraj, obs_dim]
    actions: th.Tensor              # [batch_size, len_subtraj, action_dim]

Future = History

class SubtrajBufferSample(NamedTuple):
    observations: th.Tensor     # [batch_size, obs_dim]
    actions: th.Tensor          # [batch_size, action_dim]
    history: History
    future: Future


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
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
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
            return th.tensor(array).to(self.device)
        return th.as_tensor(array).to(self.device)

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

        super(TrajectoryBuffer, self).__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device
        )
        self.max_traj_len = max_traj_len

        self.expert_dataset = expert_dataset
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
        history_marker = th.arange(-len_subtraj, 0, device=device).unsqueeze(0)
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

        self.full = True
        self.pos = self.buffer_size

    def subtraj_sample(self, batch_size: int, len_subtraj: int) -> SubtrajBufferSample:
        """
        Sample the subtrajectory of the expert data

        Note: Batch size is "Maximum" batch size
        This is due to that we only collect the subtrajectories
        whose front-rear trajectory indices are inside.

        즉, 앞뒤로 len_subtraj의 길이를 가지는 subtrajectory를 뽑았을 때, 그 index가
        전체 trajectory의 길이를 벗어나지 않는 경우에 대해서만 collect한다
        """
        low_thresh = len_subtraj - 1
        high_thresh = self.max_traj_len - len_subtraj
        timestep = np.random.randint(low=low_thresh + 1, high=high_thresh)

        # 앞뒤로 len_subtraj 만큼 잘랐을 때, index가 넘어가지 않는 친구들
        valid_indices, _ = np.nonzero(self.traj_lengths >= (timestep + len_subtraj + 1))
        # valid_indices = np.random.permutation(valid_indices)
        batch_indices = valid_indices[:batch_size]
        current_data = (
            self.observation_traj[batch_indices, timestep, :],
            self.action_traj[batch_indices, timestep, :]
        )
        current = tuple(map(self.to_torch, current_data))

        history_data = (
            self.observation_traj[batch_indices, timestep-len_subtraj:timestep, :],
            self.action_traj[batch_indices, timestep-len_subtraj:timestep]
        )
        history = History(*tuple(map(self.to_torch, history_data)))

        # 현재 것 빼고 해야하므로 +1이 붙는 것
        future_data = (
            self.observation_traj[batch_indices, timestep+1 : timestep+1+len_subtraj, :],
            self.action_traj[batch_indices, timestep+1 : timestep+1+len_subtraj, :]
        )
        future = History(*tuple(map(self.to_torch, future_data)))

        return SubtrajBufferSample(*current, history, future)

    def _get_samples(
        self, batch_inds: np.ndarray
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        raise NotImplementedError


if __name__ == "__main__":
    import gym
    env = gym.make("MountainCarContinuous-v0")
    buffer = TrajectoryBuffer(
        expert_data_path="../../expertdata/MountainCarContinuous-v0/expert_buffer-10",
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
    for i in range(100):
        z = buffer.subtraj_sample(
            batch_size=10,
            len_subtraj=100
        )