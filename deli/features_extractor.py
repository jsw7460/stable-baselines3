from typing import Union, Tuple
import torch as th
from stable_baselines3.common.torch_layers import create_mlp
from buffers import SubtrajBufferSample, TrajectoryBuffer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
from buffers import SubtrajBufferSample


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
        # assert history_latent.dim == future_latent == 2
        # flatten_input = th.cat((observations, history_latent, future_latent), dim=1)
        # return self.flatten(flatten_input)
        return self.flatten(observations)


# class TrajEmbedding(th.nn.Module):
#     """
#     Embedding network for the Trajectory Embedding
#     Use common layer in the first location
#     """
#     def __init__(
#         self,
#         observation_dim: int,
#         action_dim: int,
#         additional_dim: int = 1,
#         features_dim: int = 256,
#         latent_dim: int = 50,
#         device: Union[th.device, str] = "cuda"
#     ):
#         """
#         History라는 정보 +1 차원 추가해줌 or Future라는 정보 +1 차원 추가해줌
#             ----> Additional dim = 1
#         """
#         super(TrajEmbedding, self).__init__()
#
#         self.device = device
#         common_layer = create_mlp(
#             input_dim=observation_dim + action_dim + additional_dim,
#             output_dim=features_dim,
#             net_arch=[64, 64]
#         )
#         self.common_layer = th.nn.Sequential(*common_layer)
#
#         # After common layer, we concatenate the output of common layer to the original input
#         history_layer = create_mlp(
#             input_dim=features_dim+observation_dim+action_dim+additional_dim,
#             output_dim=latent_dim,
#             net_arch=[64, 64]
#         )
#         self.history_layer = th.nn.Sequential(*history_layer)
#
#         # After common layer, we concatenate the output of common layer to the original input
#         future_layer = create_mlp(
#             input_dim=features_dim+observation_dim+action_dim+additional_dim,
#             output_dim=latent_dim,
#             net_arch=[64, 64]
#         )
#         self.future_layer = th.nn.Sequential(*future_layer)
#
#         self.history_marker = th.ones()
#
#     def forward(self, history: th.Tensor, future: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
#         """
#         외부에서 한 번 합칠 것 이므로 여기에 넘겨줄 때는 Subtrajectory Buffer 클래스를 넘기는 것이 아니라 외부에서 계산된 텐서를 넘겨준다
#         NOTE: Batch size changed dynamically
#         History: [batch_size, len_subtraj, obs_dim + action_dim]
#         Future: [batch_size, len_subtraj, obs_dim + action_dim]
#
#         Return: ([batch_size, latent_dim], [batch_size, latent_dim]): History, Future에 대한 latent vector
#
#         b: batch size
#         l: len_subtraj
#         o: obs_dim
#         a: action_dim
#         """
#         batch_size, len_subtraj, *_ = history.size()
#
#         # 아래의 history_tensor & future_tensor: [batch, len_subtraj, obs dim + act dim + additional dim]
#         history, future = TrajectoryBuffer.timestep_marking(history, future)
#
#         common_input = th.cat((history, future), dim=1)     # [b, 2 * l, o + a + additional]
#         common_feature = self.common_layer(common_input)    # [b, 2 * l, features_dim]
#         common_feature = th.mean(common_feature, dim=1, keepdim=True)
#         common_feature = common_feature.repeat(1, len_subtraj, 1)   # [b, l, features_dim]
#
#         history_input = th.cat((common_feature, history), dim=2)    # [b, l, features_dim + o + a + additional dim]
#         future_input = th.cat((common_feature, future), dim=2)      # [b, l, features_dim + o + a + additional dim]
#
#         history_latent = self.history_layer(history_input)
#         future_latent = self.future_layer(future_input)
#
#         return history_latent, future_latent