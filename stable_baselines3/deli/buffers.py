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
G_NUM_BUFFER_REPEAT = 128


class History(NamedTuple):
    observations: th.Tensor         # [batch_size, len_subtraj, obs_dim]
    actions: th.Tensor              # [batch_size, len_subtraj, action_dim]


Future = History
STFuture = Future
LTFuture = Future


class SubtrajBufferSample(NamedTuple):
    observations: th.Tensor         # [batch_size, obs_dim]
    actions: th.Tensor              # [batch_size, action_dim]
    history: History
    future: Future


class STermSubtrajBufferSample(NamedTuple):        # Short term  futures
    observations: th.Tensor         # [batch_size, obs_dim]
    actions: th.Tensor              # [batch_size, action_dim]
    history: History
    st_future: Future               # Short Term Future


class LSTermSubtrajBufferSample(NamedTuple):        # Short term - Long term futures
    observations: th.Tensor         # [batch_size, obs_dim]
    actions: th.Tensor              # [batch_size, action_dim]
    next_observations: th.Tensor    # [batch_size, obs_dim]
    history: History
    st_future: Future               # Short Term Future
    lt_future: Future               # Long Term Future


class GoalcondBufferSample(NamedTuple):
    observations: th.Tensor         # [batch_size, obs_dim]
    next_observations: th.Tensor    # [batch_size, obs_dim]
    actions: th.Tensor              # [batch_size, action_dim]
    history: History
    goal: th.Tensor                 # [batch_size, obs_dim]


class ReplayBufferSample(NamedTuple):
    observations: th.Tensor         # [batch_size, obs_dim]
    actions: th.Tensor              # [batch_size, action_dim]


class RtgBufferSample(NamedTuple):
    observations: th.Tensor         # [batch_size, obs_dim]
    actions: th.Tensor              # [batch_size, action_dim]
    rtg: th.Tensor                  # return to go: [batch_size, ]


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    """
    # NOTE: If POMDP, give observation and action space with partially observable one

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        observation_dim: int,  # Sometimes, observation space is dict. We have to pass the specific dimension
        device: Union[th.device, str] = "cpu",
    ):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.pomdp_hidden_dim = 0       # Used for POMDP

        if observation_dim is None:
            self.observation_dim = self.obs_shape[0]
        else:
            self.observation_dim = observation_dim

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
        observation_dim: int = None,
        device: Union[th.device, str] = "cpu",
        remove_dim: list = [],
        d4rl: bool = False,
    ):
        # Load the expert dataset and set the buffer size by the size of expert dataset
        import pickle
        with open(expert_data_path+".pkl", "rb") as f:
            expert_dataset = pickle.load(f)        # Dictionary

        # Load wheather there is a hidden pomdp dimension.
        # self.pomdp_hidden_dim = expert_dataset.get("pomdp_hidden_dim", 0)
        self.pomdp_hidden_dim = 0
        self.remove_dim = remove_dim

        buffer_size = len(expert_dataset)
        max_traj_len = max([len(traj["observations"]) for traj in expert_dataset])
        super(TrajectoryBuffer, self).__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            observation_dim=observation_dim,
            action_space=action_space,
            device=device
        )
        self.max_traj_len = max_traj_len

        self.expert_dataset = expert_dataset
        self.normalizing = None
        self.reward_normalizing = None

        self.full_observation_traj = None       # Used for POMDP
        self.observation_traj = np.zeros(
            (self.buffer_size, self.max_traj_len, self.observation_dim), dtype=observation_space.dtype
        )
        self.action_traj = np.zeros(
            (self.buffer_size, self.max_traj_len, self.action_dim), dtype=action_space.dtype
        )
        self.traj_lengths = np.zeros(
            (self.buffer_size, 1), dtype=action_space.dtype
        )

        self.use_reward = False
        self.use_terminal = False
        self.reward_traj = None                 # Used if there exist reward information in the dataset
        self.terminal_traj = None               # Used if there exist terminal information in the dataset

        self.buffer_sample = None
        self.reset()

    def sample_rtg(self, batch_size, gamma=0.99) -> RtgBufferSample:
        assert self.reward_traj is not None
        batch_inds = np.random.randint(0, self.buffer_size, size=batch_size)
        timestep = np.random.randint(0, self.traj_lengths)[batch_inds].squeeze()

        observations = self.observation_traj[batch_inds, timestep, ...]
        actions = self.action_traj[batch_inds, timestep, ...]

        batch_rewards = self.reward_traj[batch_inds]

        return_to_go = np.zeros((batch_size, 1))
        for b in range(batch_size):
            weight = [gamma ** n for n in range(self.max_traj_len - timestep[b])]
            return_to_go[b] = np.sum(weight * batch_rewards[b, timestep[b]:])

        data = (
            observations,
            actions,
            return_to_go
        )
        data = tuple(map(self.to_torch, data))
        return RtgBufferSample(*data)

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
        if self.pomdp_hidden_dim > 0:
            self.full_observation_traj = np.zeros(
            shape=(self.buffer_size, self.max_traj_len, self.observation_dim + self.pomdp_hidden_dim),
            dtype=self.observation_space.dtype
        )

        use_reward = True if "rewards" in self.expert_dataset[0] else False
        if use_reward:
            self.use_reward = True
            self.reward_normalizing = 1000
            self.reward_traj = np.zeros((self.buffer_size, self.max_traj_len))
        use_terminal = True if "terminals" in self.expert_dataset[0] else False
        if use_terminal:
            self.use_terminal = True
            self.terminal_traj = np.zeros((self.buffer_size, self.max_traj_len))

        for traj_idx in range(self.buffer_size):
            traj_data = self.expert_dataset[traj_idx].copy()
            traj_len = len(traj_data["observations"])       # How many transitions in the trajectory
            self.observation_traj[traj_idx, :traj_len, ...] = traj_data["observations"]
            self.action_traj[traj_idx, :traj_len, ...] = traj_data["actions"]

            if use_reward:
                self.reward_traj[traj_idx, :traj_len] = traj_data["rewards"] / self.reward_normalizing
            if use_terminal:
                self.terminal_traj[traj_idx, :traj_len] = traj_data["terminals"]

            self.traj_lengths[traj_idx, ...] = len(traj_data["observations"])

        max_obs = np.max(self.observation_traj)
        min_obs = np.min(self.observation_traj)
        normalizing = np.max([max_obs, -min_obs])
        self.normalizing = normalizing
        self.observation_traj /= normalizing

        self.full = True
        self.pos = self.buffer_size

    # subtraj_sample: Padding 하지 않기 위해서, 앞쪽 뒤쪽에서는 sampling하지 않음. 그래서 size가 변함
    def subtraj_sample(
        self,
        batch_size: int,
        history_len_subtraj: int,
        st_future_len: int = None,      # Short term future length
        lt_future_len: int = None,      # Long term future length
        include_current: bool = False
    ) -> LSTermSubtrajBufferSample:

        if st_future_len is None:
            st_future_len = history_len_subtraj
        high_thresh = self.max_traj_len - history_len_subtraj

        # 미래 G_FUTURE_THRESH 까지의 state 중 하나를 뽑아서 goal로 설정한다. 또는 초기 state를 뽑아준다
        epsilon = np.random.uniform(0, 1)

        if epsilon < 0.5:       # 앞쪽 부분 학습 (Zero padding 하기 싫다 ~!!!!!!!!!!)
            timestep = np.random.randint(low=1, high=history_len_subtraj + 1)
            valid_indices, *_ = np.nonzero(self.traj_lengths > timestep + st_future_len + 1)

            batch_indices = valid_indices[:batch_size]
            assert len(batch_indices) > 0
            current_observed = self.observation_traj[batch_indices, timestep, :].copy()
            current_observed[..., self.remove_dim] = 0

            current_data = (
                current_observed,
                self.action_traj[batch_indices, timestep, :]
            )
            current = tuple(map(self.to_torch, current_data))

            next_observed = self.observation_traj[batch_indices, timestep+1, :].copy()
            next_observed[..., self.remove_dim] = 0

            next_observation = self.to_torch(next_observed)

            upper_bound = timestep + 1 if include_current else timestep

            history_observed = self.observation_traj[batch_indices, :upper_bound, :].copy()
            history_observed[..., self.remove_dim] = 0

            history_data = (
                history_observed,
                self.action_traj[batch_indices, :upper_bound]
            )
            history = History(*tuple(map(self.to_torch, history_data)))

            future_observed = self.observation_traj[batch_indices, timestep+1: timestep+st_future_len].copy()
            future_observed[..., self.remove_dim] = 0
            future_data = (
                future_observed,
                self.action_traj[batch_indices, timestep+1: timestep+st_future_len]
            )
            st_future = STFuture(*(tuple(map(self.to_torch, future_data))))

            # Long term observation index. Different for each batch
            traj_idx = np.vstack([np.arange(u - lt_future_len, u, dtype=np.int) for u in self.traj_lengths])
            traj_idx = traj_idx[batch_indices]

            lt_future_observation = np.take_along_axis(self.observation_traj[batch_indices], traj_idx[..., None], axis=1)
            lt_future_observation[..., self.remove_dim] = 0

            lt_future_action = np.take_along_axis(self.action_traj[batch_indices], traj_idx[..., None], axis=1)

            lt_future_data = (
                lt_future_observation,
                lt_future_action,
            )

            lt_future = LTFuture(*(tuple(map(self.to_torch, lt_future_data))))
            return LSTermSubtrajBufferSample(*current, next_observation, history, st_future, lt_future)

        else:
            low_thresh = history_len_subtraj
            # Make a batch mold
            history_len = history_len_subtraj + 1 if include_current else history_len_subtraj
            batch_current_observation = np.zeros((G_NUM_BUFFER_REPEAT, self.observation_dim))
            batch_current_action = np.zeros((G_NUM_BUFFER_REPEAT, self.action_dim))
            batch_next_observation = np.zeros((G_NUM_BUFFER_REPEAT, self.observation_dim))
            batch_history_observation = np.zeros((G_NUM_BUFFER_REPEAT, history_len, self.observation_dim))
            batch_history_action = np.zeros((G_NUM_BUFFER_REPEAT, history_len, self.action_dim))
            batch_future_observation = np.zeros((G_NUM_BUFFER_REPEAT, st_future_len, self.observation_dim))
            batch_future_action = np.zeros((G_NUM_BUFFER_REPEAT, st_future_len, self.action_dim))
            batch_lt_future_observation = None
            batch_lt_future_action = None

            if lt_future_len > 0:
                batch_lt_future_observation = np.zeros((G_NUM_BUFFER_REPEAT, lt_future_len, self.observation_dim))
                batch_lt_future_action = np.zeros((G_NUM_BUFFER_REPEAT, lt_future_len, self.action_dim))

            for batch_idx in range(G_NUM_BUFFER_REPEAT):
                timestep = np.random.randint(low=low_thresh, high=high_thresh - st_future_len - 1)
                if timestep < history_len_subtraj:
                    raise NotImplementedError

                valid_indices, *_ = np.nonzero(self.traj_lengths > timestep + st_future_len + 1)
                valid_indices = np.random.permutation(valid_indices)

                batch_indices = valid_indices[0]
                # assert len(batch_indices) > 0

                batch_current_observation[batch_idx] = self.observation_traj[batch_indices, timestep, :]
                batch_current_action[batch_idx] = self.action_traj[batch_indices, timestep, :]

                batch_next_observation[batch_idx] = self.observation_traj[batch_indices, timestep+1, :]

                upper_bound = timestep + 1 if include_current else timestep
                batch_history_observation[batch_idx] \
                    = self.observation_traj[batch_indices, timestep - history_len_subtraj:upper_bound, ...]
                batch_history_action[batch_idx] \
                    = self.action_traj[batch_indices, timestep - history_len_subtraj:upper_bound, ...]

                batch_future_observation[batch_idx] \
                    = self.observation_traj[batch_indices, timestep+1: timestep+1+st_future_len, ...]
                batch_future_action[batch_idx] \
                    = self.action_traj[batch_indices, timestep+1: timestep+1+st_future_len, ...]

                valuable_lengths = self.traj_lengths[batch_indices, :]
                traj_idx = np.vstack([np.arange(u - lt_future_len, u, dtype=np.int) for u in valuable_lengths])

                batch_lt_future_observation[batch_idx] = self.observation_traj[batch_indices, traj_idx, ...]
                batch_lt_future_action[batch_idx] = self.action_traj[batch_indices, traj_idx, ...]

            batch_current_observation[..., self.remove_dim] = 0
            current_data = (batch_current_observation, batch_current_action)
            current = tuple(map(self.to_torch, current_data))

            batch_next_observation[..., self.remove_dim] = 0
            next_data = (batch_next_observation,)
            next_step = tuple(map(self.to_torch, next_data))

            batch_history_observation[..., self.remove_dim] = 0
            history_data = (batch_history_observation, batch_history_action)
            history = History(*tuple(map(self.to_torch, history_data)))

            # batch_future_action[..., self.remove_dim] = 0            #   ???
            future_data = (batch_future_observation, batch_future_action)
            st_future = STFuture(*tuple(map(self.to_torch, future_data)))

            lt_future_data = (batch_lt_future_observation, batch_lt_future_action)
            lt_future = LTFuture(*tuple(map(self.to_torch, lt_future_data)))
            return LSTermSubtrajBufferSample(*current, *next_step, history, st_future, lt_future)

    def _get_samples(
        self, batch_inds: np.ndarray
    ):
        raise NotImplementedError

    # get_history_sample: Padding을 하기 때문에, subtrajectory의 길이가 항상 일정함
    def get_history_sample(
        self,
        batch_size: int,
        history_len: int,
        st_future_len: int = None,
    ) -> STermSubtrajBufferSample:

        batch_inds = np.random.randint(0, self.buffer_size, size=batch_size)
        timesteps = np.random.randint(0, self.traj_lengths)[batch_inds].squeeze()

        current_observations = self.observation_traj[batch_inds, timesteps, ...]
        current_actions = self.action_traj[batch_inds, timesteps, ...]

        history_observations = np.zeros((batch_size, history_len, self.observation_dim))
        history_actions = np.zeros((batch_size, history_len, self.action_dim))

        st_future_observations = np.zeros((batch_size, st_future_len, self.observation_dim))
        st_future_actions = np.zeros((batch_size, st_future_len, self.action_dim))

        for idx, batch in enumerate(batch_inds):
            timestep = timesteps[idx]
            hist_obs = self.observation_traj[batch, timestep - history_len: timestep, ...]
            hist_act = self.action_traj[batch, timestep - history_len: timestep, ...]
            cur_hist_len = len(hist_obs)        # 앞쪽에서 timestep이 골라지면, history_len보다 짧아진다.

            hist_padding_obs = np.zeros((history_len - cur_hist_len, self.observation_dim))
            hist_padding_act = np.zeros((history_len - cur_hist_len, self.action_dim))
            hist_obs = np.vstack((hist_padding_obs, hist_obs))
            hist_act = np.vstack((hist_padding_act, hist_act))

            history_observations[idx] = hist_obs
            history_actions[idx] = hist_act

            st_fut_obs = self.observation_traj[batch, timestep + 1: timestep + 1 + st_future_len, ...]
            st_fut_act = self.action_traj[batch, timestep + 1: timestep + 1 + st_future_len, ...]
            cur_st_fut_len = len(st_fut_obs)

            st_fut_padding_obs = np.zeros((st_future_len - cur_st_fut_len, self.observation_dim))
            st_fut_padding_act = np.zeros((st_future_len - cur_st_fut_len, self.action_dim))
            st_fut_obs = np.vstack((st_fut_obs, st_fut_padding_obs))
            st_fut_act = np.vstack((st_fut_act, st_fut_padding_act))

            st_future_observations[idx] = st_fut_obs
            st_future_actions[idx] = st_fut_act

        current_data = (current_observations, current_actions)
        current = tuple(map(self.to_torch, current_data))

        history_data = (history_observations, history_actions)
        history = History(*tuple(map(self.to_torch, history_data)))

        st_future_data = (st_future_observations, st_future_actions)
        st_future = Future(*tuple(map(self.to_torch, st_future_data)))

        return STermSubtrajBufferSample(*current, history, st_future)

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
        observation_dim: int = None,
        d4rl: bool = False
    ):
        super(HindsightBuffer, self).__init__(
            expert_data_path=expert_data_path,
            observation_space=observation_space,
            observation_dim=observation_dim,            # Sometimes we have to specify the observation dimension since it is dictionary, as in ant environment
            action_space=action_space,
            device=device,
            d4rl=d4rl,
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
        include_current: bool = False,
        include_longterm: bool = False,     # used for context embedding of long term goals
        remove_dim: list = []
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

            current_observed = self.observation_traj[batch_indices, timestep, :].copy()
            current_observed[:, remove_dim] = 0

            current_data = (
                current_observed,
                self.action_traj[batch_indices, timestep, :]
            )
            current = tuple(map(self.to_torch, current_data))

            upper_bound = timestep + 1 if include_current else timestep
            history_observed = self.observation_traj[batch_indices, timestep - len_subtraj:upper_bound, :].copy()
            history_observed[:, :, remove_dim] = 0

            history_data = (
                history_observed,
                self.action_traj[batch_indices, timestep - len_subtraj:upper_bound]
            )
            history = History(*tuple(map(self.to_torch, history_data)))

            goal_indices = np.random.randint(
                low=timestep,
                high=timestep + np.ones_like(self.traj_lengths[batch_indices]) * G_FUTURE_THRESH
            ).squeeze()

            goal_observed = self.observation_traj[batch_indices, goal_indices, :].copy()
            goal_observed[:, remove_dim] = 0

            goal_data = goal_observed
            goal = self.to_torch(goal_data)
            return GoalcondBufferSample(*current, history, goal)

        else:       # BATCH TRAINING
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

            batch_current_observation[..., remove_dim] = 0
            current_data = (batch_current_observation, batch_current_action)
            current = tuple(map(self.to_torch, current_data))

            batch_history_observation[..., remove_dim] = 0
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

        # Load wheather there is a hidden pomdp dimension.
        self.pomdp_hidden_dim = expert_dataset.get("pomdp_hidden_dim", 0)

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
        self.full_observations = None           # Used for POMDP
        self.n_trajectory = n_trajectory
        self.reset()

    def add(self, *args, **kwargs) -> None:
        pass

    def extend(self, *args, **kwargs) -> None:
        pass

    def reset(self) -> None:
        if self.pomdp_hidden_dim > 0:
            self.full_observations = np.zeros((self.buffer_size, self.observation_dim + self.pomdp_hidden_dim))
        self.observations = np.zeros((self.buffer_size, self.observation_dim), dtype=self.observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=self.action_space.dtype)

        top_pos = 0
        for traj in range(self.n_trajectory):
            ith_observations = self.expert_dataset["observation_trajectories"][traj]
            ith_actions = self.expert_dataset["action_trajectories"][traj]
            traj_length = len(ith_observations)

            if self.pomdp_hidden_dim > 0:
                self.full_observations[top_pos : top_pos + traj_length, ...] = np.vstack(ith_observations)

            self.observations[top_pos: top_pos + traj_length, ...] \
                = np.vstack(ith_observations)[..., :self.observation_dim]

            self.actions[top_pos: top_pos + traj_length, ...] = np.vstack(ith_actions)
            top_pos += traj_length

        self.pos = 0
        self.full = True

    def batch_sample(self, batch_size: int, remove_dim: list):
        batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size

        observed = self.observations[batch_inds, :].copy()
        observed[:, remove_dim] = 0
        data = (
            observed,
            self.actions[batch_inds, :],
        )
        return ReplayBufferSample(*tuple(map(self.to_torch, data)))

    def _get_samples(
        self, batch_inds: np.ndarray
    ):
        pass


# Used for S4RL Sampler static method
BETA_SAMPLER = th.distributions.Beta(th.FloatTensor([0.4]), th.FloatTensor([0.4]))


class S4RLBuffer(ReplayBuffer):
    def __init__(
        self,
        expert_data_path: str,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        aug_type: str = "gaussian"
    ):
        assert aug_type in [
            "gaussian_noise",
            "uniform_noise",
            "random_scaling",
            "dimension_dropout",
            "state_switch",
            "state_mixup",
            "adversarial"
        ]
        super(S4RLBuffer, self).__init__(
            expert_data_path=expert_data_path,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        self.aug_type = aug_type
        self.next_observations = np.zeros_like(self.observations)
        self.next_observations[:-1] = self.observations[1:].copy()

        super().reset()


    def batch_sample(self, batch_size: int):
        batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size

        data = (
            self.observations[batch_inds, :],
            self.actions[batch_inds, :],
            self.next_observations[batch_inds, :]
        )
        return ReplayBufferSample(*tuple(map(self.to_torch, data)))

    @classmethod
    def gaussain_noise(cls, observations: th.Tensor, **kwargs) -> th.Tensor:
        noise = th.randn_like(observations) * 3e-4
        return observations + noise

    @staticmethod
    def uniform_noise(observations: th.Tensor, **kwargs) -> th.Tensor:
        noise = th.rand(0, 3e-4)
        return observations + noise

    @staticmethod
    def random_scaling(observations: th.Tensor, **kwargs) -> th.Tensor:
        noise = th.randn_like(observations) * 3e-4
        return observations * noise

    @staticmethod
    def dimension_dropout(observations: th.Tensor, **kwargs) -> th.Tensor:
        zero_one = th.empty_like(observations).uniform_(0, 1)
        zero_one = th.bernoulli(zero_one)
        return observations * zero_one

    @staticmethod
    def state_switch(observations: th.Tensor, **kwargs) -> th.Tensor:
        # Switch random two columns
        n_dim = observations.size(1)
        indices = th.arange(n_dim)
        switching = th.randperm(n_dim)[:2]
        indices[switching[0]] = switching[1]
        indices[switching[1]] = switching[0]

        return observations[:, indices]

    @staticmethod
    def state_mixup(observations: th.Tensor, next_observations: th.Tensor, **kwargs) -> th.Tensor:
        lamb = BETA_SAMPLER.sample()
        return lamb * observations + (1 - lamb) * next_observations

    @staticmethod
    def adversarial(observations: th.Tensor, model, **kwargs):
        _observation = observations.clone().requires_grad_()
        action_pred = th.mean(model(_observation))
        action_pred.backward()
        noise = _observation.grad
        return observations + 1e-4 * noise


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
