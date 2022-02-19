from typing import Optional
from typing import Tuple

import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import create_mlp


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
        encoder_arch = [500, 500]
        encoder_net = create_mlp(state_dim + action_dim, 1000, encoder_arch)
        self.encoder = nn.Sequential(*encoder_net)

        mean_arch = [500, 500]
        mean_net = create_mlp(1000, latent_dim, mean_arch)
        self.mean = nn.Sequential(*mean_net)

        std_arch = [500, 500]
        std_net = create_mlp(1000, latent_dim, std_arch)
        self.log_std = nn.Sequential(*std_net)  # Log !!

        decoder_arch = [500, 500]
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
