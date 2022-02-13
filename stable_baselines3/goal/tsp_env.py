from typing import Union

import numpy as np
import gym
import torch as th
from gym import spaces

from stable_baselines3.common.buffers import RolloutBuffer


class TSP(gym.Env):
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        self.observation_space = spaces.Dict(
            {
                "loc": spaces.Box(low=0, high=1, shape=(n_nodes, 2)),
                "dist": spaces.Box(low=0, high=1, shape=(n_nodes, n_nodes)),
                "visited": spaces.Discrete(2),
                "cur_coord": spaces.Box(low=0, high=1, shape=(2,)),
                "timestep": spaces.Discrete(n_nodes),
            }
        )
        self.action_space = spaces.Discrete(n_nodes)


    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass


class DictGoalBuffer(RolloutBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super(DictGoalBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        assert isinstance(self.obs_shape, dict), "DictRolloutBuffer must be used with Dict obs space only"

        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        assert isinstance(self.obs_shape, dict), "DictRolloutBuffer must be used with Dict obs space only"
        self.observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            self.observations[key] = np.zeros((self.buffer_size, self.n_envs) + obs_input_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

