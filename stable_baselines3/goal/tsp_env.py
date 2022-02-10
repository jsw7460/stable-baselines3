import gym
from gym import spaces


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