from typing import Any, Dict, List, Optional, Type

import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import BasePolicy, register_policy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule


class DistributionalQNetwork(BasePolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        features_extractor: nn.Module,
        features_dim: int,
        num_atoms: int = 51,
        min_rewards: int = 0,
        max_rewards: int = 10,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        """
        Here, the Q network returns the distributions of total discounted returns for each action.
        """
        super(DistributionalQNetwork, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.num_atoms = num_atoms
        self.normalize_images = normalize_images
        action_dim = self.action_space.n
        self.atoms = th.linspace(min_rewards,  max_rewards, steps=num_atoms, device=self.device)

        q_net = create_mlp(self.features_dim, action_dim * num_atoms, self.net_arch, self.activation_fn)
        self.q_net = nn.Sequential(*q_net)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the atom probabilities of each action

        :return: The distribution of returns for each action.
        """
        y = self.q_net(self.extract_features(obs))      # [batch_size or 1, action_dim * num_atoms]
        y = y.view(-1, self.action_space.n, self.num_atoms)  # [batch_size or 1, action_dim, num_atoms]
        y = th.softmax(y, dim=-1)
        return y

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_value_distributions = self.forward(observation)       # [batch_size or 1, action_dim, num_atoms]
        # Greedy action : First we need to take expectation of distributions, which coincides with original q value.

        q_value_device = q_value_distributions.device
        self.atoms = self.atoms.to(q_value_device)

        q_values = th.sum(q_value_distributions * self.atoms, dim=-1)        # [batch_size, action_dim]
        action = q_values.argmax(dim=1).reshape(-1)
        return action

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


class C51Policy(BasePolicy):
    """
    Policy clas with Distributional Q-value network for C51

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
    """

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Schedule,
                 net_arch: Optional[List[int]] = None,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 num_atoms: int = 51,
                 features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 normalize_images: bool = True,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 ):
        super(C51Policy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs
        )
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [64, 64]

        self.num_atoms = num_atoms
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.normalize_images = normalize_images

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "num_atoms": self.num_atoms,      # Information about atom should be inserted to the distributional q network.
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self.dist_q_net, self.dist_q_net_target = None, None
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.

        Put the target network into evaluation mode.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.dist_q_net = self.make_dist_q_net()
        self.dist_q_net_target = self.make_dist_q_net()
        self.dist_q_net_target.load_state_dict(self.dist_q_net.state_dict())
        self.dist_q_net_target.set_training_mode(False)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def make_dist_q_net(self) -> DistributionalQNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return DistributionalQNetwork(**net_args).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Get the action according to the policy for a given observation.
        return self.dist_q_net._predict(obs, deterministic=deterministic)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_args["net_arch"],
                activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.dist_q_net.set_training_mode(mode)
        self.training = mode


MlpPolicy = C51Policy

register_policy("MlpPolicy", MlpPolicy)
