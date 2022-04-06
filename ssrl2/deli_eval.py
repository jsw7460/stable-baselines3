import argparse

import torch as th
import gym

from models import Deli
from models.deli import evaluate_deli

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--vae_feature_dim", type=int, default=10)
    parser.add_argument("--latent_dim", type=int, default=25)
    parser.add_argument("--additional_dim", type=int, default=1)

    args = parser.parse_args()

    env = gym.make(args.env)
    env_name = env.unwrapped.spec.id  # String. Used for save the model file.

    expert_data_path = f"/workspace/expertdata/{args.env}/expert_buffer-10"

    policy_kwargs = {
        "activation_fn": th.nn.ReLU,
        "vae_feature_dim": args.vae_feature_dim,
        "additional_dim": args.additional_dim
    }

    model_kwargs = {
        "env": env,
        "expert_data_path": expert_data_path,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
        "dropout": args.dropout,
        "tensorboard_log": f"/workspace/delilog/tensorboard/{env_name}/deli-dropout{args.dropout}-seed{args.seed}",
        "latent_dim": args.latent_dim,
        "policy_kwargs": policy_kwargs,
        "verbose": 1
    }

    model = Deli(**model_kwargs)

    # model = Deli.load_deli(f"/workspace/delilog/model/{env_name}/deli-dropout{args.dropout}-seed{args.seed}")
    reward_mean, *_ = evaluate_deli(model, env, n_eval_episodes=10)
    print("-----------------")
    print("\t reward_mean", reward_mean)
    print("-----------------")
