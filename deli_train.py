import argparse

import torch as th
import gym

from stable_baselines3 import DeliG, DeliC, DeliMG
from stable_baselines3.deli import evaluate_deli

import functools

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", type=str)
    parser.add_argument("--env_name", type=str, default="halfcheetah-medium-v2")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--vae_feature_dim", type=int, default=5)
    parser.add_argument("--latent_dim", type=int, default=25)
    parser.add_argument("--additional_dim", type=int, default=1)

    parser.add_argument("--buffer_size", type=int, default=10)
    parser.add_argument("--perturb", type=float, default=0.0)
    parser.add_argument("--context_length", type=int, default=10)

    parser.add_argument("--var", type=int, default=0)
    parser.add_argument("--grad_flow", action="store_true")

    args = parser.parse_args()

    env = gym.make(args.env_name)
    env_name = env.unwrapped.spec.id  # String. Used for save the model file.

    expert_data_path = f"/workspace/expertdata/dttrajectory/{args.env_name}"

    model_kwargs = {}
    algo = None
    if args.algo == "delig":
        algo = DeliG
    elif args.algo == "delic":
        algo = DeliC
    elif args.algo == "delimg":
        algo = DeliMG

    evaluator = functools.partial(evaluate_deli, context_length=args.context_length)
    filename_head = f"/workspace/delilog/"
    filename_tail = f"{env_name}/" \
                    f"{args.algo}" \
                    f"-context{args.context_length}" \
                    f"-grad{int(args.grad_flow)}" \
                    f"-seed{args.seed}"

    tensorboard_log = filename_head + "tensorboard/" + filename_tail

    policy_kwargs = {
        "activation_fn": th.nn.ReLU,
        "vae_feature_dim": args.vae_feature_dim,
        "additional_dim": args.additional_dim,
        "net_arch": [256, 256, 256],
    }

    model_kwargs = {
        "env": env,
        "expert_data_path": expert_data_path,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
        "dropout": args.dropout,
        "tensorboard_log": tensorboard_log,
        "latent_dim": args.latent_dim,
        "policy_kwargs": policy_kwargs,
        "verbose": 1,
        "max_traj_len": -1,
        "ent_coef": 1.0,
        "additional_dim": args.additional_dim,
        "subtraj_len": args.context_length,
        "grad_flow": args.grad_flow
    }

    model = algo(**model_kwargs)
    model.learn(1, reset_num_timesteps=False)

    random_model = algo(**model_kwargs)
    random_reward, *_ = evaluator(random_model, env, n_eval_episodes=1)

    for i in range(1000000):
        model.learn(5000, reset_num_timesteps=False)
        model.set_training_mode(False)
        reward_mean, *_ = evaluator(model, env, n_eval_episodes=10)
        normalized_mean = None
        try:
            normalized_mean = env.get_normalized_score(reward_mean) * 100
            model.logger.record("performance/rewad/normalized", normalized_mean)
        except BaseException:
            pass
        model.logger.record("performance/reward/mean", reward_mean)
        model.logger.record("performance/reward/random", random_reward)

        model._dump_logs()
        model.save(
            filename_head + "model/" + filename_tail
        )
        model.set_training_mode(True)
