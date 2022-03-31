import gym
import numpy as np
import torch as th

from deli import Deli
from vae import VAE


class HistorySampler(object):
    """
    For evaluation, we let future latent = N(0, 1) (= prior)
    And history latent is iteratively evaluated
    """
    def __init__(self, latent_dim: int, vae: VAE, device):
        self.history_observation = []
        self.history_action = []
        self.latent_dim = latent_dim
        self.vae = vae
        self.device = device

    def append(self, observation: np.ndarray, action: np.ndarray) -> None:
        """
        observation: [observation_dim]
        """
        if observation.ndim == 1:
            observation = observation[np.newaxis, :]
        self.history_observation.append(observation)
        self.history_action.append(action)

    def __len__(self):
        return len(self.history_observation)

    def get_history_latent(self):
        future_latent = np.random.randn(self.latent_dim)
        if len(self) == 0:
            history_latent = np.random.randn(self.latent_dim)
        else:
            history_obs = np.concatenate(self.history_observation, axis=0)
            history_act = np.concatenate(self.history_action, axis=0)
            history = np.concatenate((history_obs, history_act), axis=1)    # [Len_history, obs dim + act dim]
            vae_input = th.tensor(history, dtype=th.float32, device=self.device)
            history_latent, *_ = self.vae(vae_input)

            history_latent = history_latent.detach().cpu().numpy()

        return history_latent.squeeze(), future_latent.squeeze()


def evaluate_policy(
    model: Deli,
    env: gym.Env,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
):
    latent_dim = model.latent_dim
    sampler = HistorySampler(latent_dim, model.policy.vae, model.device)
    save_rewards = []
    save_episode_length = []
    device = model.device

    for i in range(n_eval_episodes):
        observation = env.reset()
        dones = False
        current_rewards = 0
        current_length = 0
        while not dones:
            current_length += 1
            # Get latent vector
            history_latent, future_latent = sampler.get_history_latent()
            latent = np.concatenate((history_latent, future_latent))
            policy_input = th.tensor(np.concatenate((th.tensor(observation), latent)), device=device).unsqueeze(0)
            # Get action and store the history transition
            action = model.policy._predict(observation=th.tensor(policy_input, device=device), deterministic=deterministic)
            action = action.detach().cpu().numpy()
            sampler.append(observation, action)

            next_observation, rewards, dones, infos = env.step(action)
            current_rewards += rewards

        save_rewards.append(current_rewards)
        save_episode_length.append(current_length)

    return np.mean(save_rewards), np.mean(save_episode_length)