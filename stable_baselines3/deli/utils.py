from typing import Union

import gym
import numpy as np
import torch as th

from .delic import DeliC
from .delig import DeliG
from .delimg import DeliMG
from .padelimg import PaDeliMG
from .features_extractor import VAE


def to_torch(array: np.ndarray, device: str, dtype: th.dtype = th.float64):
    return th.tensor(array.copy(), dtype=dtype, device=device)


class DeliGSampler(object):
    """
    For evaluation, we let future latent = N(0, 1) (= prior)
    And history latent is iteratively evaluated
    """
    def __init__(
        self,
        latent_dim: int,
        vae: VAE,
        device: str,
        context_length: int = 30,
        max_rtg=None,
        reward_normalizing: float = 1.0,

        observation_dim: int = None,
        action_dim: int = None
    ):

        self.history_observation = []
        self.history_action = []
        self.history_reward = []        # Used for return to go conditioned
        self.current_rtg = None

        self.reward_normalizing = reward_normalizing

        self.latent_dim = latent_dim
        self.vae = vae
        self.device = device
        self.context_length = context_length

        self.observation_dim = observation_dim
        self.action_dim = action_dim

        if latent_dim is not None:      # If rtg conditioned bc, latent dim is None
            self.future_latent = np.random.randn(self.latent_dim)
        self.max_rtg = max_rtg

    def append(self, observation: np.ndarray, action: np.ndarray, reward: float = None) -> None:
        """
        observation: [observation_dim]
        """
        if observation.ndim == 1:
            observation = observation[np.newaxis, :]
        self.history_observation.append(observation.copy())
        self.history_action.append(action.copy())
        if self.current_rtg is not None:
            self.history_reward.append(reward)
            self.current_rtg -= reward

    def __len__(self):
        return len(self.history_observation)

    def reset(self):
        self.history_observation = []
        self.history_action = []
        self.history_reward = []
        self.current_rtg = self.max_rtg

    def get_history_latent(self) -> th.Tensor:
        if len(self.history_observation) == 0:  # At the first state
            history_latent = th.randn(self.latent_dim, device=self.device)
            return history_latent
        else:
            history_obs = np.concatenate(self.history_observation, axis=0)[-self.context_length:, ...]
            history_act = np.concatenate(self.history_action, axis=0)[-self.context_length:, ...]
            history = np.concatenate((history_obs, history_act), axis=1)    # [Len_history, obs dim + act dim]
            vae_input = th.tensor(history, dtype=th.float32, device=self.device)
            history_latent, *_ = self.vae(vae_input)
            return history_latent.squeeze()

    def get_delig_policy_input(self, observation: np.ndarray) -> th.Tensor:     # Used also for delimg
        history_latent = self.get_history_latent()
        recon_goal = self.vae.decode_goal(latent=history_latent)
        th_observation = th.tensor(observation, device=self.device)
        policy_input = th.hstack((th_observation, recon_goal, history_latent)).unsqueeze(0)     # [1, xxx]
        return policy_input

    def get_delic_policy_input(self, observation: np.ndarray) -> th.Tensor:
        history_latent = self.get_history_latent()
        recon_latent = self.vae.decode_goal(latent=history_latent)
        th_observation = th.tensor(observation, device=self.device)
        policy_input = th.hstack((th_observation, recon_latent)).unsqueeze(0)
        return policy_input                                                                     # [1, xxx]

    def get_condbc_policy_input(self, observation: np.ndarray) -> th.Tensor:
        assert self.current_rtg is not None
        th_observation = th.tensor(observation, device=self.device)
        th_rtg = th.tensor(self.current_rtg, device=self.device).unsqueeze(0) / self.reward_normalizing
        policy_input = th.cat((th_observation, th_rtg), dim=0).unsqueeze(0)
        return policy_input                                                                     # [1, xxx]

    def get_padelimg_policy_input(self, observation: np.ndarray) -> th.Tensor:
        th_observation = th.tensor(observation, device=self.device)
        if len(self.history_observation) == 0:
            history_observation = np.zeros((self.context_length, self.observation_dim))
        else:
            history_observation = np.vstack(self.history_observation)[-self.context_length:, ...]

        cur_hist_len = len(history_observation)

        hist_padding_obs = np.zeros((self.context_length - cur_hist_len, self.observation_dim))
        history = np.vstack((hist_padding_obs, history_observation))

        history = th.tensor(history, device=self.device)
        history = th.flatten(history, start_dim=0)

        history_latent = self.get_history_latent()

        history = th.cat((history, th_observation))

        policy_input = th.cat((history, history_latent.detach())).unsqueeze(0)
        return policy_input


def evaluate_deli(
    model: Union[DeliG, DeliC, DeliMG, PaDeliMG],
    env: gym.Env,
    n_eval_episodes: int = 10,
    context_length: int = 30,
    deterministic: bool = True,
    normalizing_factor: float = None,
    max_rtg=None,
    **kwargs
):
    if normalizing_factor is None:
        normalizing_factor = model.replay_buffer.normalizing

    latent_dim = model.latent_dim
    vae = model.vae
    reward_normalizing = model.replay_buffer.reward_normalizing

    sampler = DeliGSampler(
        latent_dim,
        vae,
        model.device,
        context_length,
        max_rtg,
        reward_normalizing,
        model.observation_dim,
        model.action_dim
    )
    save_rewards = []
    save_episode_length = []

    algo = type(model).__name__
    for i in range(n_eval_episodes):
        sampler.reset()
        observation = env.reset()
        observation /= normalizing_factor
        dones = False
        current_rewards = 0
        current_length = 0
        while not dones:
            current_length += 1
            policy_input = None
            # Get policy input

            if algo == "DeliG" or algo == "DeliMG":
                policy_input = sampler.get_delig_policy_input(observation)
            elif algo == "DeliC":
                policy_input = sampler.get_delic_policy_input(observation)
            elif algo == "CondSACBC":
                policy_input = sampler.get_condbc_policy_input(observation)
            elif algo == "PaDeliMG":
                policy_input = sampler.get_padelimg_policy_input(observation)

            action = model.policy._predict(policy_input, deterministic=deterministic)
            action = action.detach().cpu().numpy()
            action = action.squeeze()

            if action.ndim == 0:
                action = np.expand_dims(action, axis=0)

            next_observation, rewards, dones, infos = env.step(action)
            sampler.append(observation, action[np.newaxis, ...], rewards)
            current_rewards += rewards

            observation = (next_observation.copy() / normalizing_factor)

        save_rewards.append(current_rewards)
        save_episode_length.append(current_length)

    return np.mean(save_rewards), np.mean(save_episode_length)
