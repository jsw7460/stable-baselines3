import argparse

import torch as th
import gym

from models import SACBC

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--vae_feature_dim", type=int, default=10)
    parser.add_argument("--latent_dim", type=int, default=25)
    parser.add_argument("--additional_dim", type=int, default=1)

    args = parser.parse_args()

    env = gym.make(args.env)
    env_name = env.unwrapped.spec.id  # String. Used for save the model file.

    expert_data_path = f"/workspace/expertdata/{args.env}/expert_buffer-10"

    policy_kwargs = {
        "activation_fn": th.nn.ReLU,
    }

    model_kwargs = {
        "policy": "MlpPolicy",
        "env": env,
        "expert_data_path": expert_data_path,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
        "tensorboard_log": f"/workspace/delilog/{env_name}/bc/tensorboard/seed{args.seed}",
        "policy_kwargs": policy_kwargs,
        "verbose": 1
    }

    model = SACBC(**model_kwargs)
    for i in range(1000000):
        model.learn(500, reset_num_timesteps=False)
        model._dump_logs()
        model.save(f"/workspace/delilog/{env_name}/bc/model/seed{args.seed}.zip")