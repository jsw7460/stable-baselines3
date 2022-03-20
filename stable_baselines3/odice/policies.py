import torch as th

from stable_baselines3.common.buffers import RolloutBufferSamples
from stable_baselines3.common.torch_layers import (
    create_mlp,
)

from functools import partial

class DonskerMinimizer(th.nn.Module):
    def __init__(
        self,
        observation_space,
    ):
        super(DonskerMinimizer, self).__init__()
        self.observation_space = observation_space
        obs_dim = observation_space.shape[0]
        net_arch = [256, 256]
        layer = create_mlp(obs_dim, 1, net_arch)
        self.layer = th.nn.Sequential(*layer)

    def forward(self, observation):
        observation = observation.float()
        return self.layer(observation)


from stable_baselines3.sac.policies import Actor


def get_kl_divergence(
    donsker: th.nn.Module,
    replay_data: RolloutBufferSamples,
    target_actor: Actor,                 # Policy class가 아니고 actor class를 받는게 더 좋아보인다
    behavior_actor: Actor,
    gamma: float,
    policy_no_grad: bool = False
):
    """
    Target Policy, Behavior Policy 모두 Gaussian 분포를 따른다고 항상 가정하겠다.
    안그러면 못한다.
    """
    # Compute v(s')
    v_s_current = donsker(replay_data.observations)
    v_s_next = donsker(replay_data.next_observations)

    # Compute the mean and log standard deviations
    with th.set_grad_enabled(not policy_no_grad):
        t_mean, t_log_std, _ = target_actor.get_action_dist_params(replay_data.observations)
        b_mean, b_log_std, _ = behavior_actor.get_action_dist_params(replay_data.observations)

    # Compute the variances
    t_var = th.exp(t_log_std) ** 2
    b_var = th.exp(b_log_std) ** 2

    # Compute the product of component of variance. This is equivalent to product of the diagonal of covariance matrix.
    # This is needed to compute the Radon Nikodym Derivative of two Gaussian distributions.
    t_var_prod = th.prod(t_var, dim=1, keepdim=True)
    b_var_prod = th.prod(b_var, dim=1, keepdim=True)

    # Compute (x-mean).T @ inverse(covariance) @ (x-mean)
    # Since covariance is diagonal, so inverse is just reciprocal of diagonal matrix
    t_mean_dist = th.sum((1 / t_var) * ((replay_data.actions - t_mean) ** 2), dim=1, keepdim=True)
    b_mean_dist = th.sum((1 / b_var) * ((replay_data.actions - b_mean) ** 2), dim=1, keepdim=True)


    # Compute the Radon Nikodym Derivative(RND) between the two Gaussian probability measures.
    log_rnd = 0.5 * th.log(b_var_prod / t_var_prod) - 0.5 * (t_mean_dist - b_mean_dist)
    rnd = th.exp(log_rnd)
    rnd = th.clamp(rnd, 0, 10.0)
    # rnd.clamp_(0, 10)

    bellman_v_s = gamma * v_s_next * rnd
    negative_approx_kl_divergence \
        = th.log(th.mean(th.exp(v_s_current - bellman_v_s))) - (1 - gamma) * th.mean(v_s_current)

    if policy_no_grad:      # For donsker loss
        return negative_approx_kl_divergence, (t_var_prod, b_var_prod, rnd)
    else:
        return -negative_approx_kl_divergence

get_donsker_loss = partial(get_kl_divergence, policy_no_grad=True)
get_approx_kl = partial(get_kl_divergence, policy_no_grad=False)