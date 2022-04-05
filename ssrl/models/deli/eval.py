import gym
import numpy as np
import torch as th

from .deli import Deli
from ..delig import DeliG, DeliG3
from .features_extractor import VAE


class DeliSampler(object):
    """
    For evaluation, we let future latent = N(0, 1) (= prior)
    And history latent is iteratively evaluated
    """
    def __init__(self, latent_dim: int, vae: VAE, device, context_length: int = 30,):
        self.history_observation = []
        self.history_action = []
        self.latent_dim = latent_dim
        self.vae = vae
        self.device = device
        self.context_length = context_length

        self.future_latent = np.random.randn(self.latent_dim)

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
        if len(self) == 0:  # At the first state
            history_latent = np.random.randn(self.latent_dim)
        else:
            history_obs = np.concatenate(self.history_observation, axis=0)[-self.context_length:, ...]
            history_act = np.concatenate(self.history_action, axis=0)[-self.context_length:, ...]
            history = np.concatenate((history_obs, history_act), axis=1)    # [Len_history, obs dim + act dim]
            vae_input = th.tensor(history, dtype=th.float32, device=self.device)
            history_latent, *_ = self.vae(vae_input)

            history_latent = history_latent.detach().cpu().numpy()

        return history_latent.squeeze(), self.future_latent.squeeze()


class DeligSampler(object):
    """
    For evaluation, we let future latent = N(0, 1) (= prior)
    And history latent is iteratively evaluated
    """
    def __init__(self, latent_dim: int, vae: VAE, device, context_length: int = 30,):
        self.history_observation = []
        self.history_action = []
        self.latent_dim = latent_dim
        self.vae = vae
        self.device = device
        self.context_length = context_length

        self.future_latent = np.random.randn(self.latent_dim)

    def reset(self):
        self.history_observation = []
        self.history_action = []

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
        if len(self) == 0:  # At the first state
            history_latent = np.random.randn(self.latent_dim)
            goal_latent = np.random.randn(self.latent_dim)
        else:
            history_obs = np.concatenate(self.history_observation, axis=0)[-self.context_length:, ...]
            history_act = np.concatenate(self.history_action, axis=0)[-self.context_length:, ...]
            history = np.concatenate((history_obs, history_act), axis=1)

            # vae_input: [Len_history, obs dim + action_dim]
            vae_input = th.tensor(history, dtype=th.float32, device=self.device)
            history_latent, goal_latent, *_ = self.vae(vae_input)

            history_latent = history_latent.detach().cpu().numpy()
            goal_latent = goal_latent.detach().cpu().numpy()

        return history_latent.squeeze(), goal_latent.squeeze()


class DeliG3Sampler(object):
    """
    For evaluation, we let future latent = N(0, 1) (= prior)
    And history latent is iteratively evaluated
    """
    def __init__(self, latent_dim: int, vae: VAE, device, context_length: int = 30,):
        self.history_observation = []
        self.history_action = []
        self.latent_dim = latent_dim
        self.vae = vae
        self.device = device
        self.context_length = context_length

        self.future_latent = np.random.randn(self.latent_dim)

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

    def reset(self):
        self.history_observation = []
        self.history_action = []

    def get_history_latent(self):
        # print("LENGTH", len(self.history_observation))
        if len(self.history_observation) == 0:  # At the first state
            history_latent = np.random.randn(self.latent_dim)
            # history_latent = history_latent[np.newaxis, :]
            return history_latent
        else:
            history_obs = np.concatenate(self.history_observation, axis=0)[-self.context_length:, ...]
            history_act = np.concatenate(self.history_action, axis=0)[-self.context_length:, ...]
            history = np.concatenate((history_obs, history_act), axis=1)    # [Len_history, obs dim + act dim]
            vae_input = th.tensor(history, dtype=th.float32, device=self.device)
            history_latent, *_ = self.vae(vae_input)

            history_latent = history_latent.detach().cpu().numpy()
            return history_latent.squeeze()


def evaluate_deli(
    model: Deli,
    env: gym.Env,
    n_eval_episodes: int = 10,
    context_length: int = 30,
    deterministic: bool = True,
):
    latent_dim = model.latent_dim
    sampler = DeliSampler(latent_dim, model.policy.vae, model.device, context_length)
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
            action = action.squeeze()
            next_observation, rewards, dones, infos = env.step(action)
            current_rewards += rewards

            observation = next_observation

        save_rewards.append(current_rewards)
        save_episode_length.append(current_length)

    return np.mean(save_rewards), np.mean(save_episode_length)


def evaluate_delig(
    model: DeliG,
    env: gym.Env,
    n_eval_episodes: int = 10,
    context_length: int = 30,
    deterministic: bool = True,
):
    latent_dim = model.latent_dim
    sampler = DeligSampler(latent_dim, model.policy.vae, model.device, context_length)
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
            history_latent, goal_latent = sampler.get_history_latent()
            latent = np.concatenate((history_latent, goal_latent))
            # policy_input = th.tensor(np.concatenate((th.tensor(observation), latent)), device=device).unsqueeze(0)

            policy_input = th.tensor(np.hstack((observation, latent)), device=device).unsqueeze(0)
            # Get action and store the history transition
            action = model.policy._predict(observation=th.tensor(policy_input, device=device), deterministic=deterministic)
            action = action.detach().cpu().numpy()
            sampler.append(observation, action)
            action = action.squeeze()
            next_observation, rewards, dones, infos = env.step(action)
            current_rewards += rewards

            observation = next_observation

        save_rewards.append(current_rewards)
        save_episode_length.append(current_length)

    return np.mean(save_rewards), np.mean(save_episode_length)


def evaluate_delig3(
    model: DeliG3,
    env: gym.Env,
    n_eval_episodes: int = 10,
    context_length: int = 30,
    deterministic: bool = True,
):
    normalizing_factor = model.replay_buffer.normalizing
    latent_dim = model.latent_dim
    sampler = DeliG3Sampler(latent_dim, model.vae, model.device, context_length)
    save_rewards = []
    save_episode_length = []
    device = model.device

    for i in range(n_eval_episodes):
        sampler.reset()
        observation = env.reset()
        observation /= normalizing_factor
        dones = False
        current_rewards = 0
        current_length = 0
        while not dones:
            current_length += 1

            # Get latent vector
            history_latent = sampler.get_history_latent()
            recon_goal = model.vae.decode_goal(latent=th.tensor(history_latent, device=device, dtype=th.float32)).squeeze().cpu().detach().numpy()
            np_input = np.hstack((observation, recon_goal, history_latent))
            policy_input = th.tensor(np_input, device=device).unsqueeze(0)

            # Get action and store the history transition
            action = model.policy._predict(observation=th.tensor(policy_input, device=device), deterministic=deterministic)
            action = action.detach().cpu().numpy()
            sampler.append(observation, action)
            action = action.squeeze()

            if action.ndim == 0:
                action = np.expand_dims(action, axis=0)

            next_observation, rewards, dones, infos = env.step(action)
            current_rewards += rewards

            observation = next_observation / normalizing_factor

        save_rewards.append(current_rewards)
        save_episode_length.append(current_length)

    return np.mean(save_rewards), np.mean(save_episode_length)