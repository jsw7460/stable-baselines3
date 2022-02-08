from typing import Any, Dict, List, Optional, Type, Union
from typing import Tuple

import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, register_policy
from stable_baselines3.common.preprocessing import get_action_dim, get_flattened_obs_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import Schedule


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
        encoder_arch = [300, 300]
        encoder_net = create_mlp(state_dim + action_dim, 750, encoder_arch)       # 750: Same with original BCQ code
        self.encoder = nn.Sequential(*encoder_net)

        mean_arch = [200, 200]
        mean_net = create_mlp(750, latent_dim, mean_arch)
        self.mean = nn.Sequential(*mean_net)

        std_arch = [200, 200]
        std_net = create_mlp(750, latent_dim, std_arch)
        self.log_std = nn.Sequential(*std_net)      # Log !!

        decoder_arch = [200, 200]
        decoder_net = create_mlp(state_dim + latent_dim, action_dim, decoder_arch, squash_output=True)
        self.decoder = nn.Sequential(*decoder_net)

        self.latent_dim = latent_dim
        self.max_action = max_action

    def forward(self, state: th.Tensor, action: th.Tensor) -> Tuple:
        """
        state: [batch size, state dimension]
        action: [batch size, action dimension]

        Return: (reconstructed action, mean, log_std)
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


class Actor(BasePolicy):
    """
        Actor network (policy) for TD3.

        :param observation_space: Obervation space
        :param action_space: Action space
        :param net_arch: Network architecture
        :param features_extractor: Network to extract features
            (a CNN when using images, a nn.Flatten() layer otherwise)
        :param features_dim: Number of features
        :param activation_fn: Activation function
        :param normalize_images: Whether to normalize images or not,
             dividing by 255.0 (True by default)
        """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            net_arch: List[int],
            features_extractor: nn.Module,
            features_dim: int,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,

            perturbation: float = 0.05,
    ):
        super(Actor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        action_dim = get_action_dim(self.action_space)
        state_dim = get_flattened_obs_dim(self.observation_space)

        self.autoencoder = VariationalAutoEncoder(
            state_dim,
            action_dim,
            latent_dim=100,
            max_action=self.action_space.high[0],
        )
        self.max_action = self.autoencoder.max_action

        perturb_net = create_mlp(features_dim + action_dim, action_dim, net_arch=[64, 64], squash_output=True)
        self.perturb_net = nn.Sequential(*perturb_net)
        self.perturbation = perturbation

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        state --> [state, generator's action] --> Apply perturbation
        """
        action = self.autoencoder.decode(obs, device=self.device)
        return self.perturb_action(obs, action)

    def perturb_action(self, obs: th.Tensor, action: th.Tensor) -> th.Tensor:
        """
        Action: generator's reconstructed action.
        """
        perturb_input = th.cat([obs, action], dim=1).float()
        perturb = self.perturb_net(perturb_input)

        return (action + perturb * self.perturbation).clamp_(-self.max_action, self.max_action)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self.forward(observation)


class BCQPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for TD3.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super(BCQPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )
        # Default network architecture, from the original paper
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [400, 300]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        # Create actor and target
        # the features extractor should not be shared
        self.actor = self.make_actor(features_extractor=None)
        self.actor_target = self.make_actor(features_extractor=None)
        # Initialize the target to have the same weights as the actor
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Critic target should not share the features extactor with critic
            # but it can share it with the actor target as actor and critic are sharing
            # the same features_extractor too
            # NOTE: as a result the effective poliak (soft-copy) coefficient for the features extractor
            # will be 2 * tau instead of tau (updated one time with the actor, a second time with the critic)
            self.critic_target = self.make_critic(features_extractor=self.actor_target.features_extractor)
        else:
            # Create new features extractor for each network
            self.critic = self.make_critic(features_extractor=None)
            self.critic_target = self.make_critic(features_extractor=None)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(self.critic.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        # Target networks should always be in eval mode
        self.actor_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                share_features_extractor=self.share_features_extractor,
            )
        )
        return data

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.

        return self.actor(observation)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


MlpPolicy = BCQPolicy

register_policy("MlpPolicy", MlpPolicy)
