from stable_baselines3 import TRPO
import gym

if __name__ == "__main__":
    env = gym.make("HalfCheetah-v2")
    model = TRPO("MlpPolicy", env=env)

    model.learn(10000)

