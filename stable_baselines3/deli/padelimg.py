from collections import deque
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch as th

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from .buffers import TrajectoryBuffer
from .features_extractor import NextStatePredictor
from .policies import MSEBCPolicy
from ..deli.features_extractor import HistoryVAE

DEQUE = partial(deque, maxlen=100)


class PaDeliMG(OffPolicyAlgorithm):
    """
    Twin Delayed DDPG (TD3)
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
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = 1,
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

        without_exploration: bool = True,
        gumbel_ensemble: bool = False,
        gumbel_temperature: float = 0.5,
        expert_data_path: str = None,  # If evaluation, None
        dropout: float = 0.1,
        additional_dim: int = 1,
        vae_feature_dim: int = 256,
        latent_dim: int = 128,
        subtraj_len: int = 10,
        use_st_future: bool = True,
        **kwargs
    ):
        assert without_exploration
        policy_kwargs["dropout"] = dropout
        policy_kwargs["history_len"] = subtraj_len
        policy_kwargs["latent_dim"] = latent_dim
        super(PaDeliMG, self).__init__(
            "MSEBCPolicy",
            env,
            MSEBCPolicy,
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
            gumbel_temperature=gumbel_temperature,
            dropout=dropout
        )

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise
        self.dropout = dropout
        self.additional_dim = additional_dim
        self.vae_feature_dim = vae_feature_dim
        self.latent_dim = latent_dim
        self.subtraj_len = subtraj_len
        self.use_st_future = use_st_future

        self.kl_losses, self.recon_losses = DEQUE(), DEQUE()
        self.history_mues, self.history_stds = DEQUE(), DEQUE()
        self.model_pred_losses = DEQUE()
        self.actor_losses = DEQUE()

        if expert_data_path is not None:
            self.replay_buffer = TrajectoryBuffer(
                expert_data_path=expert_data_path,
                observation_space=env.observation_space,
                observation_dim=kwargs.get("observation_dim", None),
                action_space=env.action_space,
                device=self.device,
            )
            self.normalizing = self.replay_buffer.normalizing       # Normalizing factor of observations.

        self.observation_dim = kwargs.get("observation_dim", None)     # For some env, we give explicit observation dim
        if self.observation_dim is None:
            self.observation_dim = self.observation_space.shape[0]

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(PaDeliMG, self)._setup_model()
        self._create_aliases()

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.vae = HistoryVAE(
            state_dim=self.observation_dim,
            action_dim=self.action_space.shape[0],
            feature_dim=self.vae_feature_dim,
            latent_dim=self.latent_dim,
            recon_dim=self.observation_dim,
            additional_dim=self.additional_dim
        ).to(self.device)
        self.vae.optimizer = th.optim.Adam(self.vae.parameters(), lr=1e-4)

        self.model_predictor = NextStatePredictor(self.observation_dim, self.action_dim).to(self.device)
        self.model_predictor.optimizer = th.optim.Adam(self.model_predictor.parameters(), lr=1e-4)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer])

        self._n_updates += 1
        # Sample replay buffer

        replay_data = self.replay_buffer.get_history_sample(
            batch_size=batch_size,
            history_len=self.subtraj_len,
            st_future_len=10,
        )

        # Note ---- Start: Learn action predict model
        next_state_pred = self.model_predictor(replay_data.observations, replay_data.actions)
        next_state = replay_data.st_future.observations[:, 0, :]
        state_pred_loss = th.mean((next_state_pred - next_state) ** 2)

        self.model_pred_losses.append(state_pred_loss.item())
        self.model_predictor.zero_grad()
        state_pred_loss.backward()
        self.model_predictor.optimizer.step()
        # Note ---- End: Learn action predict model

        # Define the input data by concatenating the ingradients.
        history_tensor = th.cat((replay_data.history.observations, replay_data.history.actions), dim=2)
        history_latent, history_stat = self.vae(history_tensor)

        # NOTE ---- Start: Latent vector KL-loss
        history_mu, history_log_std = history_stat
        history_std = th.exp(history_log_std)
        history_kl_loss = -0.5 * (1 + th.log(history_std.pow(2)) - history_mu.pow(2) - history_std.pow(2)).mean()

        kl_loss = history_kl_loss.mean()

        # Save logs
        self.kl_losses.append(kl_loss.item())
        self.history_mues.append(history_mu.mean().item())
        self.history_stds.append(history_std.mean().item())
        # NOTE ---- End: Latent vector KL-loss

        # NOTE ---- Start: Goal encoder-decoder reconstruction loss
        # Before defining the goal reconstruction loss, we have to define the goal according to the
        # derivative of self.action_predictor with respect to the future state
        fut_obs = replay_data.st_future.observations
        fut_act = replay_data.st_future.actions
        max_grad_indices = self.model_predictor.highest_grads(fut_obs, fut_act)
        n = max_grad_indices.size(0)  # n == Batch size; dynamically chagnes

        goals = fut_obs[th.arange(n), max_grad_indices]
        goal_recon = self.vae.decode_goal(history_tensor)
        goal_recon_loss = th.mean((goal_recon - goals) ** 2)
        self.recon_losses.append(goal_recon_loss.item())

        vae_loss = kl_loss + goal_recon_loss
        self.vae.zero_grad()
        vae_loss.backward()
        self.vae.optimizer.step()
        # NOTE ---- End: Goal encoder-decoder reconstruction loss

        history_observation = th.flatten(replay_data.history.observations, start_dim=1)
        history_observation = th.cat((history_observation, replay_data.observations), dim=1)
        policy_input = th.cat((history_observation, history_latent.detach()), dim=1)
        action_pred = self.actor(policy_input)
        bc_loss = th.mean((action_pred - replay_data.actions) ** 2)

        # Optimize the actor
        self.actor_losses.append(bc_loss.item())
        self.actor.optimizer.zero_grad()
        bc_loss.backward()
        self.actor.optimizer.step()

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", np.mean(self.actor_losses))
        self.logger.record("train/vae_kl_loss", np.mean(self.kl_losses))
        self.logger.record("train/vae_mean", np.mean(self.history_mues))
        self.logger.record("train/vae_stds", np.mean(self.history_stds))
        self.logger.record("train/model_pred_loss", np.mean(self.model_pred_losses))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "PaDeliMG",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(PaDeliMG, self).learn(
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
        return super(PaDeliMG, self)._excluded_save_params() + ["actor"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "vae", "model_predictor"]
        return state_dicts, []
