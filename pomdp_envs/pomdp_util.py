import numpy as np

import gym
from gym.spaces import Discrete, Box


class RemoveDim(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        remove_dim: list,
    ):
        super(RemoveDim, self).__init__(env)
        self.env = env
        self.remove_dim = remove_dim

    def observation(self, observation):
        observation[self.remove_dim] = 0
        return observation


if __name__ == "__main__":
    env = gym.make("HalfCheetah-v2")
    env = RemoveDim(env, [0, 2])

    z = env.reset()
    print(z)
