import gym
import numpy as np
import torch as th

from .histbc import HistBC


class HistorySampler(object):
    def __init__(self, context_length, normalizing_factor: float = 1.0, device: str = "cuda:0"):
        self.history_obs = []
        self.context_length = context_length
        self.normalizing_factor = normalizing_factor
        self.device = device

    def __len__(self):
        return len(self.history_obs)

    def sample(self, to_torch: bool = True):
        assert len(self.history_obs) > 0
        np_input = np.vstack(self.history_obs.copy())[-self.context_length: ]
        np_input = np_input[np.newaxis, :, :]
        if to_torch:
            return th.tensor(np_input, dtype=th.float, device=self.device)
        else:
            return np_input

    def add(self, observation: np.ndarray):
        self.history_obs.append(observation / self.normalizing_factor)

    def reset(self):
        self.history_obs = []


def evaluate_histbc(
    model: HistBC,
    env: gym.Env,
    n_eval_episodes: int = 10,
    context_length: int = 30,
    deterministic: bool = True,
):
    device = model.device
    sampler = HistorySampler(context_length, model.replay_buffer.normalizing, device=device)
    save_rewards = []
    save_episode_length = []

    for i in range(n_eval_episodes):
        sampler.reset()
        observation = env.reset()
        sampler.add(observation)
        dones = False
        current_rewards = 0
        current_length = 0
        while not dones:
            current_length += 1
            # Get history vector
            history = sampler.sample(to_torch=True)

            # Get action and store the history transition
            action = model.policy._predict(history, deterministic=deterministic)
            action = action.detach().cpu().numpy()
            action = action.squeeze()

            if action.ndim == 0:
                action = np.expand_dims(action, axis=0)

            next_observation, rewards, dones, infos = env.step(action)
            sampler.add(next_observation.copy())
            current_rewards += rewards

        save_rewards.append(current_rewards)
        save_episode_length.append(current_length)

    return np.mean(save_rewards), np.mean(save_episode_length)