from typing import Tuple
from buffers import TrajectoryBuffer
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import create_mlp


class VAE(nn.Module):
    """
    말이 VAE이지, 우선 학습은 Policy를 높이는 방향으로 학습한다. NOTE: 이것의 variation으로 reconstruction 하는 것도 가능하다.
    PEARL, MAESN, VariBAD 등에서 Latent Vector를 학습 시키는 방법과 동일하다
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        feature_dim: int,
        latent_dim: int,
        additional_dim: int,
    ):
        """
        additional_dim: Trajectory의 앞부분인지 뒷부분인지에 대한 추가 정보를 넣어줘야 함. 그에 대한 Dimension
        """
        super(VAE, self).__init__()

        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.optimizer = None

        # Front Trajectory를 embedding하는 것과 rear Trajectory를 embedding하는 network의 architecture는 동일하게 사용
        # Common_net: used as a shared encoder of trajectory.
        # We only use the separated mean-std structure
        encoder_arch = [32, 32]
        encoder_arch = create_mlp(state_dim + action_dim + additional_dim, feature_dim, encoder_arch, dropout=0.0)
        self.encoder = th.nn.Sequential(*encoder_arch)

        # NOTE: Wheter use squashed output?
        # NOTE: Input dim[0] = feature_dim + state_dim + action_dim + 1
        # 이렇게 되는 이유는, common에서 뽑은 feature에다가 원래 인자로 받는 history를 concat해서 layer 에다가 넣어줄 것이기 때문
        mean_arch = [32, 32]
        mean_arch = create_mlp(feature_dim + state_dim + action_dim + additional_dim, latent_dim, mean_arch)
        self.history_mu = th.nn.Sequential(*mean_arch)
        self.future_mu = th.nn.Sequential(*mean_arch)

        log_std_arch = [32, 32]
        log_std_arch = create_mlp(feature_dim + state_dim + action_dim + additional_dim, latent_dim, log_std_arch)
        self.history_log_std = th.nn.Sequential(*log_std_arch)
        self.future_log_std = th.nn.Sequential(*log_std_arch)

    def forward(self, history: th.Tensor, future: th.Tensor = None) -> Tuple:
        """
        :param history: [batch_size, len_subtraj, obs_dim + action_dim]
        :param future: [batch_size, len_subtraj, obs_dim + action_dim]
        future may be none, especially when evaluation.

        return: Two latent vectors, corresponding to history and future
        """
        if history.dim() == 1:
            history = history.unsqueeze(0).unsqueeze(1)
        elif history.dim() == 2:
            history = history.unsqueeze(0)
        else:
            pass

        # 아래의 history_tensor & future_tensor: [batch, len_subtraj, obs dim + act dim + additional dim]
        history, future = TrajectoryBuffer.timestep_marking(history, future, history.device)
        history_stat, future_stat = self.encode(history, future)
        history_mu, history_log_std = history_stat
        history_latent = self.get_latent_vec(history_mu, history_log_std)
        if future is None:
            return history_latent, history_mu, history_log_std

        future_mu, future_log_std = future_stat
        future_latent = self.get_latent_vec(future_mu, future_log_std)
        return history_latent, future_latent, (history_mu, history_log_std), (future_mu, future_log_std)

    @staticmethod
    def get_latent_vec(mu: th.Tensor, log_std: th.Tensor) -> th.Tensor:
        std = th.exp(log_std)
        latent_vector = mu + std * th.randn_like(std)
        return latent_vector

    def encode(self, history: th.Tensor, future: th.Tensor = None) -> Tuple:
        """
        NOTE: Input history and future should be preprocessed before here, inside forward function.
        NOTE: For example, use timestep_marking not here, but in the forward method.
        NOTE: Thus, additional dim is included in history and future

        history: [batch_size, len_subtraj, obs_dim + action_dim + additional_dim]
        future: [batch_size, len_subtraj, obs_dim + action_dim + additional_dim]

        future may be none, especially when evaluation.
        """
        batch_size, len_subtraj, *_ = history.size()

        if future is None:
            future = history
        encoder_input = th.cat((history, future), dim=1)
        common_embedding = self.encoder(encoder_input)  # [batch, 2 * len_subtraj, feature_dim]
        common_embedding = th.mean(common_embedding, dim=1, keepdim=True)
        common_embedding = common_embedding.repeat(1, len_subtraj, 1)  # [batch, len_subtraj, feature_dim]

        history_input = th.cat((common_embedding, history), dim=2)
        future_input = th.cat((common_embedding, future), dim=2)

        history_mean = self.history_mu(history_input)
        history_mean = th.mean(history_mean, dim=1)
        history_log_std = th.clamp(self.history_log_std(history_input), -4, 15)
        history_log_std = th.mean(history_log_std, dim=1)

        if future is None:
            return history_mean, history_log_std

        future_mean = self.future_mu(future_input)
        future_mean = th.mean(future_mean, dim=1)
        future_log_std = th.clamp(self.future_log_std(future_input), -4, 15)
        future_log_std = th.mean(future_log_std, dim=1)

        return (history_mean, history_log_std), (future_mean, future_log_std)
