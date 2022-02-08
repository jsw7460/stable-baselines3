from typing import Optional, List

import gym
import torch as th

from stable_baselines3.common.torch_layers import create_mlp


class Discriminator(th.nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        learning_rate: float = 5e-4,
    ):
        """
        Deals with continuous action space
        """
        super(Discriminator, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space

        self.learning_rate = learning_rate

        action_dim = self.action_space.shape[0]
        observation_dim = self.observation_space.shape[0]
        self.layer = th.nn.Sequential(
            th.nn.Linear(observation_dim + action_dim, 64),
            th.nn.ReLU(inplace=True),
            th.nn.Linear(128, 128),
            th.nn.ReLU(inplace=True),
            th.nn.Linear(128, 128),
            th.nn.ReLU(inplace=True),
            th.nn.Linear(64, 1),
            th.nn.Sigmoid()
        )

        self.optimizer = th.optim.Adam(self.layer.parameters(), lr=learning_rate, betas=(0.5, 0.99))

    def forward(self, observations: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        :param observations: [batch_size, observation_dim]
        :param actions: [batch_size, aciton_dim]
        :return: th.Tensor
        """
        cat = th.cat([observations, actions], dim=1)
        return self.layer(cat)


class AugCQLDiscriminator(th.nn.Module):
    def __init__(self,
         observation_space: gym.spaces.Space,
         action_space: gym.spaces.Space,
         learning_rate: float = 5e-4,
         net_arch: Optional[List] = None,
         use_reward_and_next_state: bool = True,
        ):
        """
        :param use_reward_and_next_state: If true, the conditional GAN made, with condition on
        reward and the next_state
        """
        super(AugCQLDiscriminator, self).__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.net_arch = net_arch
        self.use_reward_and_next_state = use_reward_and_next_state

        if net_arch is None:
            self.net_arch = [256, 256]

        input_dim = observation_space.shape[0] + action_space.shape[0]
        if use_reward_and_next_state:       # Conditional gan.
            input_dim += (1 + observation_space.shape[0])

        disc_net = create_mlp(input_dim, 1, net_arch=self.net_arch, squash_output=False)
        disc_net.append(th.nn.Sigmoid())
        self.disc_net = th.nn.Sequential(*disc_net)

    def forward(
            self,
            observation: th.Tensor,     # [batch_size, observation_dim]
            action: th.Tensor,          # [batch_size, action_dim]
            reward: Optional[th.Tensor] = None,         # [batch_size, 1]
            next_state: Optional[th.Tensor] = None      # [batch_size, observation_dim]
    ):
        if self.use_reward_and_next_state:
            assert reward is not None, "Discriminator should be conditioned on reward"
            assert next_state is not None, "Discriminator should be conditioned on next_state"
            layer_input = th.cat([observation, action, reward, next_state], dim=1)
            return self.disc_net(layer_input)
        else:
            layer_input = th.cat([observation, action])
            return self.disc_net(layer_input)


class AugCQLGenerator(th.nn.Module):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 latent_dim: int = 100,
                 learning_rate: float = 5e-4,
                 net_arch: Optional[List] = None,
                 use_reward_and_next_state: bool = True,
                 latent_dist: th.distributions = th.distributions.Normal(0, 1),
                 ):
        super(AugCQLGenerator, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.net_arch = net_arch
        self.use_reward_and_next_state = use_reward_and_next_state
        self.latent_dist = latent_dist

        if net_arch is None:
            self.net_arch = [256, 256]

        input_dim = latent_dim
        if use_reward_and_next_state:       # Conditional gan.
            input_dim += (1 + observation_space.shape[0])
        output_dim = observation_space.shape[0] + action_space.shape[0]
        gen_net = create_mlp(input_dim, output_dim, net_arch=self.net_arch, squash_output=False)
        gen_net.append()
        self.gen_net = th.nn.Sequential(*gen_net)

    def get_latent_vector(self, batch_size: int):
        return self.latent_dist.sample(sample_shape = (batch_size, 1))

    def forward(
            self,
            latent_vector: th.Tensor,
            reward: Optional[th.Tensor] = None,
            next_state: Optional[th.Tensor] = None
    ):
        if self.use_reward_and_next_state:
            assert reward is not None, "Discriminator should be conditioned on reward"
            assert next_state is not None, "Discriminator should be conditioned on next_state"
            layer_input = th.cat([latent_vector, reward, next_state], dim=1)
            return self.gen_net(layer_input)
        else:
            layer_input = latent_vector
            return self.gen_net(layer_input)

if __name__ == "__main__":
    import gym
    _env = gym.make("HalfCheetah-v2")

    _disc = AugCQLDiscriminator(_env.observation_space, _env.action_space, use_reward_and_next_state=True)

    _lat = th.distributions.Normal(0, 1)
    print(_lat.sample(sample_shape=(128, 1)).size())

