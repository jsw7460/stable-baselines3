import gym
import torch as th

from stable_baselines3 import TRPO

if __name__ == "__main__":



    env = gym.make("Pendulum-v1")
    model = TRPO("MlpPolicy", env, verbose=1)

    model.learn(10000000)