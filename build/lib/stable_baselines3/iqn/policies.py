from typing import Any, Dict, List, Optional, Type, Tuple

import numpy as np
import gym
import torch as th
from torch import nn
from torch.distributions.normal import Normal

from stable_baselines3.common.policies import BasePolicy, register_policy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule


class DistortionMeasures(object):
    @classmethod
    def cvar_(cls, tau: th.Tensor, eta: float = 0.5) -> th.Tensor:
        """
        :param tau: [batch_size, num_tau]
        """
        return eta * tau

    @classmethod
    def pow_(cls, tau: th.Tensor, eta: float = -2) -> th.Tensor:
        """
        :param tau: [batch_size, num_tau]
        """
        if eta >= 0:
            return th.exp(1 / (1 + np.abs(eta)) * th.log(tau))
        else:
            return 1 - th.exp(1 / (1 + np.abs(eta)) * th.log(1 - tau))

    @classmethod
    def wang_(cls, tau: th.Tensor, eta: float = -0.75) -> th.Tensor:
        """
        :param tau: [batch_size, num_tau]
        """
        normal_distribution = Normal(0, 1)
        return normal_distribution.cdf(normal_distribution.icdf(tau) + eta)

    @classmethod
    def cpw_(cls, tau: th.Tensor, eta: float = 0.71) -> th.Tensor:
        """
        :param tau: [batch_size, num_tau]
        """
        return (tau ** eta) / th.exp(th.log(tau ** eta + (1 - tau) ** eta) / eta)


class QuantileNetwork(BasePolicy):
    """
    Quantile network for the Implicit Quantile Network.

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            features_extractor: nn.Module,
            features_dim: int,
            distortion_measure: str = "cvar",
            n_tau: int = 50,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
    ):
        super(QuantileNetwork, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.n_cos = 32     # We fix n = 64 in the paper
        self.n_tau = n_tau
        self.distortion_measure = distortion_measure
        self.distortion_measure_fn = self._get_distortion_measure(self.distortion_measure)
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.normalize_images = normalize_images

        action_dim = self.action_space.n  # number of actions

        # In the IQN, they change the ffn by doing a hadamard product with a linear map phi.
        # We use exactly same notation with the paper.
        # In the paper, the dimension d (SEE THE DOMAIN OF THE FUNCTION f in the paper)
        # is net_arch[-1] in our case. : d = net_arch[-1]

        # phi : A network before f
        psi = create_mlp(self.features_dim, self.net_arch[-1], self.net_arch[: -1], self.activation_fn)
        f = create_mlp(self.net_arch[-1], action_dim, [self.net_arch[-1]], self.activation_fn)

        self.psi = nn.Sequential(*psi)
        self.f = nn.Sequential(*f)

        self.pi_matrix = th.FloatTensor([np.pi * i for i in range(self.n_cos)]).view(1, 1, self.n_cos).to(self.device)
        self.cos_embedding = nn.Sequential(
            nn.Linear(self.n_cos, self.net_arch[-1]),
            nn.ReLU()
        )
        self.cos_embedding = nn.Linear(self.n_cos, self.net_arch[-1])

    def _get_distortion_measure(self, distortion_measure: str):
        if distortion_measure.startswith("cv") or distortion_measure.startswith("CV"):
            return DistortionMeasures.cvar_
        elif distortion_measure.startswith("po") or distortion_measure.startswith("PO"):
            return DistortionMeasures.pow_
        elif distortion_measure.startswith("wa") or distortion_measure.startswith("WA"):
            return DistortionMeasures.wang_
        elif distortion_measure.startswith("cp") or distortion_measure.startswith("CP"):
            return DistortionMeasures.cpw_
        else:
            raise KeyError("Unsupported distortion measure")

    def generate_sample_points_of_tau(self, size: Tuple[int, ...]) -> Tuple[th.Tensor, ...]:
        tau = th.rand(size=size, device=self.device)
        beta_tau = self.distortion_measure_fn(tau)
        return beta_tau, tau

    def _transform_tau(self, tau: th.Tensor) -> th.Tensor:
        """
        This function returns the cos(pi * i * tau) in the paper, for the batch case.
        :param tau: A tau or beta(tau) in the paper [batch_size, num_tau]
        :return: Transformed tau into n_cos dimension
        """
        tau.unsqueeze_(2)
        tr_tau = tau.repeat(1, 1, self.n_cos)
        try:
            tr_tau = th.cos(tr_tau * self.pi_matrix)
        except RuntimeError:
            self.pi_matrix = self.pi_matrix.to(self.device)
            tr_tau = th.cos(tr_tau * self.pi_matrix)
        return tr_tau       # [batch_size, num_tau, n_cos = 64 in the paper.]

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, ...]:
        """
        Predict the quantiles from given observation

        Here, we generate a uniform sampling tau and apply the risk distortion measure.

        In the IQN paper, they change the feed forward network by
        doing a hadamard product with phi(tau)

        :param obs: observation
        :return: Estimated Quantiles
        """

        obs = obs.float()
        batch_size = obs.size(0)

        beta_tau, tau = self.generate_sample_points_of_tau((batch_size, self.n_tau))
        transformed_tau = self._transform_tau(beta_tau)      # [batch_size, num_tau, n_cos]
        phi_tau = self.cos_embedding(transformed_tau)  # [batch_size, num_tau, d]

        # psi_x = self.psi(self.features_extractor(obs))      # [batch_size, d]
        features = self.features_extractor(obs)
        psi_x = self.psi(features)
        # Do the Hadamard product
        psi_x.unsqueeze_(1)
        # Get the input of the function f
        hadamard_product = psi_x * phi_tau      # [batch_size, num_tau, d]

        result = self.f(hadamard_product)   # [batch_size, num_tau, action_dim]

        # Return the tau. This will be used to calculate the Huber loss function.
        return result, tau

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        """
        :return: Predicted actions
        """
        q_values, _ = self.forward(obs)         # q_values: [batch_size, num_tau, action_dim]
        q_values = q_values.mean(dim=1)                 # [batch_size, action_dim]
        action = q_values.argmax(dim=1).reshape(-1)     # [batch_size]
        return action

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                n_tau=self.n_tau,
                distortion_measure=self.distortion_measure,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
                n_cos=self.n_cos
            )
        )
        return data


class IQNPolicy(BasePolicy):
    """
    Policy class with quantile and target networks for Implicit Quantile Network.


    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param n_tau: A sampling number of tau.
    :param distortion_measure: A distortion measure. Beta in the paper
    :param net_arch: The specification of the network architecture.
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
                 n_tau: int = 50,
                 distortion_measure: str = "cvar",
                 net_arch: Optional[List[int]] = None,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 normalize_images: bool = True,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None):
        super(IQNPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.n_tau = n_tau
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.normalize_images = normalize_images
        self.distortion_measure = distortion_measure
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_tau": self.n_tau,
            "distortion_measure": self.distortion_measure,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images
        }

        self.quantile_net, self.quantile_net_target = None, None
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.

        Put the target network into evaluation mode.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.quantile_net = self.make_quantile_net()
        self.quantile_net_target = self.make_quantile_net()
        self.quantile_net_target.load_state_dict(self.quantile_net.state_dict())
        self.quantile_net_target.set_training_mode(False)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def make_quantile_net(self) -> QuantileNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return QuantileNetwork(**net_args).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        """
        Here we generate tau and transform it according to the risk distortion measure.
        """

        return self.quantile_net._predict(obs, deterministic=deterministic)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                n_tau=self.net_args["n_tau"],
                distortion_measure=self.net_args["distortion_measure"],
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
        self.quantile_net.set_training_mode(mode)
        self.training = mode


MlpPolicy = IQNPolicy


class CnnPolicy(IQNPolicy):
    """
        Policy class for DQN when using images as input.

        :param observation_space: Observation space
        :param action_space: Action space
        :param lr_schedule: Learning rate schedule (could be constant)
        :param net_arch: The specification of the policy and value networks.
        :param activation_fn: Activation function
        :param features_extractor_class: Features extractor to use.
        :param normalize_images: Whether to normalize images or not,
             dividing by 255.0 (True by default)
        :param optimizer_class: The optimizer to use,
            ``th.optim.Adam`` by default
        :param optimizer_kwargs: Additional keyword arguments,
            excluding the learning rate, to pass to the optimizer
        """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_tau: int = 50,
            distortion_measure: str = "cvar",
    ):
        super(CnnPolicy, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_tau=n_tau,
            distortion_measure=distortion_measure,
        )


register_policy("MlpPolicy", MlpPolicy)
register_policy("CnnPolicy", CnnPolicy)