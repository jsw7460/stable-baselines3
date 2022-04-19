from typing import Tuple

import torch as th
from .buffers import TrajectoryBuffer
from stable_baselines3.common.torch_layers import create_mlp
from torch import nn


class ActionPredictor(th.nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int
    ):
        super(ActionPredictor, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        net_arch = [256, 256]
        predictor_net = create_mlp(observation_dim, action_dim, net_arch, squash_output=True)
        self.predictor = th.nn.Sequential(*predictor_net)

        self.optimizer = None

    def forward(self, observation: th.Tensor):
        return self.predictor(observation)

    def highest_grads(self, observation: th.Tensor):
        """
        observations: [batch_size, len_trajectory, observation_dim]
        Given observation, this indices of observations which has the highest derivate of predictor network.
        """
        _observation = observation.clone().requires_grad_()
        action_pred = th.mean(self.predictor(_observation))
        action_pred.backward()
        grad_norm = th.norm(_observation.grad, dim=2)

        _, max_grad_indices = th.max(grad_norm, dim=1)
        return max_grad_indices


class NextStatePredictor(th.nn.Module):
    def __init__(self, observation_dim: int, action_dim: int):
        super(NextStatePredictor, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        net_arch = [256, 256]
        predictor_net = create_mlp(observation_dim + action_dim, observation_dim, net_arch, squash_output=True)
        self.predictor = th.nn.Sequential(*predictor_net)

        self.optimizer = None

    def forward(self, observation: th.Tensor, action: th.Tensor) -> th.Tensor:
        net_input = th.cat((observation, action), dim=-1)
        return self.predictor(net_input)

    def highest_grads(self, observation: th.Tensor, action: th.Tensor):
        _observation = observation.clone().requires_grad_()
        _action = action.clone().requires_grad_()
        _net_input = th.cat((_observation, _action), dim=-1)
        pred = th.mean(self.predictor(_net_input))
        pred.backward()
        grad_norm = th.norm(_action.grad, dim=2)

        _, max_grad_indices = th.max(grad_norm, dim=1)
        return max_grad_indices



class CondBCExtractor(th.nn.Module):
    def __init__(self, observation_dim, additional_dim):
        """
        Latent dim: Latent vector dimension in below "TrajEmbedding" class
        Features dim: Observation dim + 2 * latent_dim. 여기가 실제 pi (policy)에 들어가는 부분으로 보임
        """
        super(CondBCExtractor, self).__init__()
        self.flatten = th.nn.Flatten()

        # features_dim = Policy의 Input. 여기서는 observation + conditional info 임
        self._features_dim = observation_dim + additional_dim
        self.latent_dim = additional_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(
        self,
        observations: th.Tensor,
    ) -> th.Tensor:
        return self.flatten(observations)


class DeliGExtractor(th.nn.Module):
    def __init__(self, observation_dim, latent_dim):
        """
        Latent dim: Latent vector dimension in below "TrajEmbedding" class
        Features dim: Observation dim + 2 * latent_dim. 여기가 실제 pi (policy)에 들어가는 부분으로 보임
        """
        super(DeliGExtractor, self).__init__()
        self.flatten = th.nn.Flatten()

        # features_dim = Policy의 Input. 여기서는 observation + goal + history latent임
        self._features_dim = observation_dim * 2 + latent_dim
        self.latent_dim = latent_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(
        self,
        observations: th.Tensor,
    ) -> th.Tensor:
        # assert history_latent.dim == future_latent == 2
        # flatten_input = th.cat((observations, history_latent, future_latent), dim=1)
        # return self.flatten(flatten_input)
        return self.flatten(observations)


class DeliCExtractor(th.nn.Module):
    def __init__(self, observation_dim, latent_dim):
        """
        Latent dim: Latent vector dimension in below "TrajEmbedding" class
        Features dim: Observation dim + 2 * latent_dim. 여기가 실제 pi (policy)에 들어가는 부분으로 보임
        """
        super(DeliCExtractor, self).__init__()
        self.flatten = th.nn.Flatten()

        # features_dim = Policy의 Input. 여기서는 observation + goal + history latent임
        self._features_dim = observation_dim + latent_dim * 2
        self.latent_dim = latent_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(
        self,
        observations: th.Tensor,
    ) -> th.Tensor:
        return self.flatten(observations)


class TrajFlattenExtractor(th.nn.Module):
    def __init__(self, observation_dim, latent_dim):
        """
        Latent dim: Latent vector dimension in below "TrajEmbedding" class
        Features dim: Observation dim + 2 * latent_dim. 여기가 실제 pi (policy)에 들어가는 부분으로 보임
        """
        super(TrajFlattenExtractor, self).__init__()
        self.flatten = th.nn.Flatten()
        self._features_dim = observation_dim + latent_dim * 2
        self.latent_dim = latent_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(
        self,
        observations: th.Tensor,
        # history_latent: th.Tensor,
        # future_latent: th.Tensor
    ) -> th.Tensor:
        return self.flatten(observations)


class VAE(nn.Module):
    """
    말이 VAE이지, 우선 학습은 Policy를 높이는 방향으로 학습한다. NOTE: 이것의 variation으로 reconstruction 하는 것도 가능하다.
    PEARL, MAESN, VariBAD 등에서 Latent Vector를 학습 시키는 방법과 동일하다.

    History, Future trajectory 두 경로를 받아서 각각의 latent vector 생성
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
        encoder_arch = [16, 16]
        encoder_arch = create_mlp(state_dim + action_dim + additional_dim, feature_dim, encoder_arch, dropout=0.1)
        self.encoder = th.nn.Sequential(*encoder_arch)

        # NOTE: Wheter use squashed output?
        # NOTE: Input dim[0] = feature_dim + state_dim + action_dim + 1
        # 이렇게 되는 이유는, common에서 뽑은 feature에다가 원래 인자로 받는 history를 concat해서 layer 에다가 넣어줄 것이기 때문
        mean_arch = [64, 64]
        mean_arch = create_mlp(feature_dim + state_dim + action_dim + additional_dim, latent_dim, mean_arch)
        self.history_mu = th.nn.Sequential(*mean_arch)

        mean_arch = [64, 64]
        mean_arch = create_mlp(feature_dim + state_dim + action_dim + additional_dim, latent_dim, mean_arch)
        self.future_mu = th.nn.Sequential(*mean_arch)

        log_std_arch = [128, 128]
        log_std_arch = create_mlp(feature_dim + state_dim + action_dim + additional_dim, latent_dim, log_std_arch)
        self.history_log_std = th.nn.Sequential(*log_std_arch)

        log_std_arch = [128, 128]
        log_std_arch = create_mlp(feature_dim + state_dim + action_dim + additional_dim, latent_dim, log_std_arch)
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


class GoalVAE(th.nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            feature_dim: int,
            latent_dim: int,
            additional_dim: int,
    ):
        """
        History와 Goal을 받아서
        History에 대한 latent와 Goal에 대한 latent를 만들어냄
        """
        super(GoalVAE, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.additional_dim = additional_dim
        self.optimizer = None

        # Front Trajectory를 embedding하는 것과 rear Trajectory를 embedding하는 network의 architecture는 동일하게 사용
        # Common_net: used as a shared encoder of trajectory.
        # We only use the separated mean-std structure
        history_encoder_arch = [16, 16]
        history_encoder_arch \
            = create_mlp(state_dim + action_dim + additional_dim, feature_dim, history_encoder_arch, dropout=0.1)
        self.history_encoder = th.nn.Sequential(*history_encoder_arch)

        goal_encoder_arch = [256, 256]
        goal_encoder_arch \
            = create_mlp(state_dim + action_dim + additional_dim, feature_dim , goal_encoder_arch, dropout=0.1)
        self.goal_encoder = th.nn.Sequential(*goal_encoder_arch)

        mean_arch = [128, 128]
        mean_arch = create_mlp(feature_dim, latent_dim, mean_arch, dropout=0.1)
        self.history_mu = th.nn.Sequential(*mean_arch)

        mean_arch = [16, 16]
        mean_arch = create_mlp(feature_dim, latent_dim, mean_arch, dropout=0.1)
        self.goal_mu = th.nn.Sequential(*mean_arch)

        log_std_arch = [128, 128]
        log_std_arch = create_mlp(feature_dim, latent_dim, log_std_arch, dropout=0.1)
        self.history_log_std = th.nn.Sequential(*log_std_arch)

        log_std_arch = [16, 16]
        log_std_arch = create_mlp(feature_dim, latent_dim, log_std_arch, dropout=0.1)
        self.goal_log_std = th.nn.Sequential(*log_std_arch)

        goal_decoder_arch = [16, 16]
        goal_decoder_arch = create_mlp(latent_dim, state_dim, goal_decoder_arch, dropout=0.1)
        self.goal_decoder = th.nn.Sequential(*goal_decoder_arch)

    def forward(self, history: th.Tensor) -> Tuple:
        if history.dim() == 1:
            history = history.unsqueeze(0).unsqueeze(1)
        elif history.dim() == 2:
            history = history.unsqueeze(0)
        else:
            pass

        history, *_ = TrajectoryBuffer.timestep_marking(history, None, history.device)

        history_stat, goal_stat = self.encode(history)
        history_mu, history_log_std = history_stat
        goal_mu, goal_log_std = goal_stat

        history_latent = VAE.get_latent_vec(history_mu, history_log_std)
        goal_latent = VAE.get_latent_vec(goal_mu, goal_log_std)

        return history_latent, goal_latent, (history_mu, history_log_std), (goal_mu, goal_log_std)

    def encode(self, history: th.Tensor, encode_goal: bool = True) -> Tuple:
        """
        NOTE: Input history should be preprocessed before here, inside forward function.

        history: [batch_size, len_subtraj, obs_dim + action_dim + additional_dim]
        """
        history_embedding = self.history_encoder(history)
        history_embedding = th.mean(history_embedding, dim=1)
        history_mu = self.history_mu(history_embedding)
        history_log_std = self.history_log_std(history_embedding)

        if not encode_goal:
            return history_mu, history_log_std

        goal_embedding = self.history_encoder(history)  # goal embedding도 Input은 history이다. 나중에 goal이랑 recon loss 계산됨
        # assert th.isnan(goal_embedding).sum() == 0
        goal_embedding = th.mean(goal_embedding, dim=1)
        goal_mu = self.goal_mu(goal_embedding)
        goal_log_std = self.goal_log_std(goal_embedding)

        return (history_mu, history_log_std), (goal_mu, goal_log_std)

    def decode_goal(self, history: th.Tensor) -> th.Tensor:
        """
        history: [batch_size, len_subtraj, obs_dim + action_dim]
        NOTE: Before the additional information is attatched to history
        """
        history, *_ = TrajectoryBuffer.timestep_marking(history, None, history.device)
        history_mu, history_log_std = self.encode(history, encode_goal=False)
        history_latent = VAE.get_latent_vec(history_mu, history_log_std)
        recon_goal = self.goal_decoder(history_latent)
        return recon_goal


class HistoryVAE(th.nn.Module):
    """
    History를 받아서 history에 대한 latent value를 만들어냄
    3번째 variation에 사용 (DeliG3)
    """
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            feature_dim: int,
            latent_dim: int,
            recon_dim: int,
            additional_dim: int,
            squash_output: bool = True,
    ):
        super(HistoryVAE, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.recon_dim = recon_dim
        self.additional_dim = additional_dim

        history_encoder_arch = [256, 256]
        history_encoder_arch \
            = create_mlp(state_dim + action_dim + additional_dim, feature_dim, history_encoder_arch, dropout=0.1)
        self.history_encoder = th.nn.Sequential(*history_encoder_arch)

        mean_arch = [256, 256]
        mean_arch = create_mlp(feature_dim, latent_dim, mean_arch, dropout=0.1)
        self.history_mu = th.nn.Sequential(*mean_arch)

        log_std_arch = [256, 256]
        log_std_arch = create_mlp(feature_dim, latent_dim, log_std_arch, dropout=0.1)
        self.history_log_std = th.nn.Sequential(*log_std_arch)

        goal_decoder_arch = [256, 256]
        goal_decoder_arch \
            = create_mlp(latent_dim, recon_dim, goal_decoder_arch, dropout=0.1, squash_output=squash_output)
        self.goal_decoder = th.nn.Sequential(*goal_decoder_arch)

        self.optimizer = None

    def forward(self, history: th.Tensor) -> Tuple:
        if history.dim() == 1:
            history = history.unsqueeze(0).unsqueeze(1)
        elif history.dim() == 2:
            history = history.unsqueeze(0)
        else:
            pass

        history, *_ = TrajectoryBuffer.timestep_marking(history, None, history.device)
        history_mu, history_log_std = self.encode(history)
        history_latent = VAE.get_latent_vec(history_mu, history_log_std)        # [batch_size, latent_dim]
        return history_latent, (history_mu, history_log_std)

    def encode(self, history: th.Tensor) -> Tuple:
        """
        NOTE: Input history should be preprocessed before here, inside forward function.
        history: [batch_size, len_subtraj, obs_dim + action_dim + additional_dim]
        """
        history_embedding = self.history_encoder(history)
        history_embedding = th.mean(history_embedding, dim=1)
        history_mu = self.history_mu(history_embedding)
        history_log_std = self.history_log_std(history_embedding)

        return history_mu, history_log_std

    def decode_goal(self, history: th.Tensor = None, latent: th.Tensor = None) -> th.Tensor:
        """
        history: [batch_size, len_subtraj, obs_dim + action_dim]
        NOTE: Before the additional information is attatched to history
        """
        if latent is not None:
            history_latent = latent
        else:
            history, *_ = TrajectoryBuffer.timestep_marking(history, None, history.device)
            history_mu, history_log_std = self.encode(history)
            history_latent = VAE.get_latent_vec(history_mu, history_log_std)

        recon_goal = self.goal_decoder(history_latent)
        return recon_goal