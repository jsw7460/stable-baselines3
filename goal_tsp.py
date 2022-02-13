import torch as th
from stable_baselines3 import DQN

from stable_baselines3.goal.tsp_env import TSP
import pickle


if __name__ == "__main__":
    with open("./goaldata", "rb") as f:
        offline_dataset = pickle.load(f)

    env = TSP(20)
    model = DQN("MultiInputPolicy", env=env, without_exploration=True)
    print(model.replay_buffer)
    exit()
    model.register_handmade_dataset(offline_dataset)
