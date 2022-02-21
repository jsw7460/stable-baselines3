import argparse
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.preprocessing import get_action_dim, get_flattened_obs_dim
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.td3.policies import TD3Policy


class VariationalAutoEncoder(nn.Module):
    """
    Given (s, a) --> Reconstruct a
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        latent_dim,
        max_action,
    ):
        super(VariationalAutoEncoder, self).__init__()
        encoder_arch = [16, 16]
        encoder_net = create_mlp(state_dim + action_dim, 50, encoder_arch)
        self.encoder = nn.Sequential(*encoder_net)

        mean_arch = [16, 16]
        mean_net = create_mlp(50, latent_dim, mean_arch)
        self.mean = nn.Sequential(*mean_net)

        std_arch = [16, 16]
        std_net = create_mlp(50, latent_dim, std_arch)
        self.log_std = nn.Sequential(*std_net)      # Log !!

        decoder_arch = [16, 16]
        decoder_net = create_mlp(state_dim + latent_dim, action_dim, decoder_arch, squash_output=True)
        self.decoder = nn.Sequential(*decoder_net)

        self.latent_dim = latent_dim
        self.max_action = max_action

    def forward(self, state: th.Tensor, action: th.Tensor) -> Tuple:
        """
        state: [batch size, state dimension]
        action: [batch size, action dimension]

        Return: Tuple, (reconstructed action, mean, log_std)
        """

        mean, log_std = self.encode(state, action)
        # log_std.clamp_(-4, 15)      # For the stability of numerical computation  이거 코드가 이미 encode안에 포함되어있다.
        std = th.exp(log_std)

        latent_vector = mean + std * th.randn_like(std)     # [batch_size, latent_dim]
        reconstructed_action = self.decode(state, latent_vector, mean.device)
        return reconstructed_action, mean, log_std

    def encode(self, state: th.Tensor, action: th.Tensor) -> Tuple:
        """
        Return: mean, log_std
        """
        encoder_input = th.cat([state, action], dim=1).float()
        y = self.encoder(encoder_input)

        mean = self.mean(y)
        log_std = th.clamp(self.log_std(y), -4, 15)  # Clamp for stability

        return mean, log_std

    def decode(self, state: th.Tensor, latent_vec: Optional[th.Tensor] = None, device: th.device = "cuda:0") -> th.Tensor:
        if latent_vec is None:      # Used to Sample action from "next" states. See algorithm in the BCQ paper.
            batch_size = state.size(0)
            with th.no_grad():
                latent_vec = th.randn((batch_size, self.latent_dim), device=device, dtype=th.float32)
                latent_vec.clamp_(-self.max_action, +self.max_action)
        decoder_input = th.cat([state, latent_vec], dim=1).float()
        action = self.decoder(decoder_input)
        return self.max_action * action     # range: [-max_action, max_action]



class MIN(OffPolicyAlgorithm):
    """
    Minimalist approach for offline reinforcement learning.
    The algorithm based on TD3.

    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
            self,
            policy: Union[str, Type[TD3Policy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 1e-3,
            buffer_size: int = 1_000_000,  # 1e6
            learning_starts: int = 100,
            batch_size: int = 100,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
            gradient_steps: int = -1,
            action_noise: Optional[ActionNoise] = None,
            replay_buffer_class: Optional[ReplayBuffer] = None,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
            policy_delay: int = 2,
            target_policy_noise: float = 0.2,
            target_noise_clip: float = 0.5,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,

            without_exploration: bool = False,
            gumbel_ensemble: bool = False,
            gumbel_temperature: float = 0.5,
            alpha: Union[float, str] = 2.5,
    ):

        super(MIN, self).__init__(
            policy,
            env,
            TD3Policy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box),

            without_exploration=without_exploration,
            gumbel_ensemble=gumbel_ensemble,
            gumbel_temperature=gumbel_temperature
        )

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        self.behavior_cloning_criterion = th.nn.MSELoss()
        self.alpha = alpha

        self.vae = None

        self.test_step = 0

        if _init_setup_model:
            self._setup_model()

        # If we adjust alpha automatically
        if isinstance(self.alpha, str) and self.alpha.startswith("auto"):
            state_dim = get_flattened_obs_dim(self.observation_space)
            action_dim = get_action_dim(self.action_space)
            self.vae = VariationalAutoEncoder(
                state_dim,
                action_dim,
                100,
                self.action_space.high[0],
            ).to(self.device)
            self.vae_optimizer = th.optim.Adam(self.vae.parameters(), lr=1e-4)

    def _setup_model(self) -> None:
        super(MIN, self)._setup_model()
        self._create_aliases()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def get_action_variance(self) -> th.Tensor:
        """
        observations: [Batch_size, observation_dim]
        return: [batch_size, 1]
        """
        batch_size = 100
        replay_data = self.replay_buffer.sample(batch_size)
        observations = replay_data.observations

        tile_observations = th.repeat_interleave(observations, 10, 0)
        # noise = tile_observations.clone().data.normal_(0, 0.01)
        # tile_observations = tile_observations + noise

        tile_ae_action = self.vae.decode(tile_observations)
        tile_ae_action = tile_ae_action.reshape(batch_size, 10, -1)  # [batch_size, 10, action_dim]

        action_var = th.var(tile_ae_action, dim=1)
        action_var = th.mean(action_var, dim=1, keepdim=True)

        return action_var  # [batch_size, 1]

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        if self.vae is not None:
            vae_losses = []

        self._n_updates += 1
        # Sample replay buffer
        replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

        # Start: train autoencoder
        reconstructed_action, mean, log_std = self.vae(replay_data.observations, replay_data.actions)
        std = th.exp(log_std)
        # ae_kl_loss = th.log(1 / std) + (std ** 2 + mean ** 2) / 2 - 0.5
        ae_kl_loss = th.tensor([0.0])
        autoencoder_loss = th.mean((reconstructed_action - replay_data.actions) ** 2) + th.mean(ae_kl_loss)

        self.vae.zero_grad()
        autoencoder_loss.backward()
        self.vae_optimizer.step()
        vae_losses.append(autoencoder_loss.item())

        # End: train autoencoder
        self.test_step += 1
        self.logger.record("train/vae_loss", np.mean(vae_losses))
        if self.test_step % 1_000 == 0:
            print("VAE LOSS", np.mean(vae_losses))

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "MIN",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(MIN, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _excluded_save_params(self) -> List[str]:
        return super(MIN, self)._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env_name", type=str, default="halfcheetah")
    parser.add_argument("--degree", type=str, default="medium")

    args = parser.parse_args()

    env = gym.make(f'{args.env_name}-{args.degree}-v2')
    env_name = env.unwrapped.spec.id        # String. Used for save the model file.

    model = MIN("MlpPolicy", env=env, learning_starts=0, without_exploration=True, alpha="auto", gradient_steps=1)

    for epoch in range(1_000):
        model.learn(1_000, reset_num_timesteps=False)
        z = model.get_action_variance()
        print("DEGREE", args.degree)
        print("VARIANCE", th.mean(z) * 100)
        print("\n\n")
