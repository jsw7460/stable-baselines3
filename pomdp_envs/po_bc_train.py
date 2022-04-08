import argparse
import sys

import torch as th

from pomdp_halfcheetah import HalfCheetahMass
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.append("..")
from ssrl.models import SACBC, HistBC
from ssrl.models.bc import evaluate_histbc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env_name", type=str, default="HalfCheetahMass")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--buffer_size", type=int, default=10)
    parser.add_argument("--perturb", type=float, default=0.0)
    parser.add_argument("--context_length", type=int, default=0)

    args = parser.parse_args()

    env = HalfCheetahMass(use_gravity=False)
    env_name = args.env_name

    expert_data_path \
        = f"/workspace/expertdata/{args.env_name}/expert_buffer-{args.buffer_size}-perturb{args.perturb}"

    policy_kwargs = {
        "activation_fn": th.nn.ReLU,
    }

    model_kwargs = {}
    if args.context_length > 0:
        algo = HistBC
        algo_name = "histbc"
        evaluator = evaluate_histbc
        model_kwargs.update({"context_length": args.context_length})
    else:
        algo = SACBC
        algo_name = "bc"
        evaluator = evaluate_policy

    tensorboard_log = f"/workspace/delilog/tensorboard/" \
                      f"{env_name}/" \
                      f"{algo_name}" \
                      f"-buffer{args.buffer_size}" \
                      f"-perturb{args.perturb}" \
                      f"-context{args.context_length}" \
                      f"-seed{args.seed}"

    model_kwargs.update({
        "policy": "MlpPolicy",
        "env": env,
        "expert_data_path": expert_data_path,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
        "tensorboard_log": tensorboard_log,
        "policy_kwargs": policy_kwargs,
        "verbose": 1,
        "ent_coef": 1,
    })

    model = algo(**model_kwargs)

    for i in range(1000000):
        model.learn(5000, reset_num_timesteps=False)

        reward_mean, reward_std = evaluator(model, model.env)
        # normalized_reward_mean = env.get_normalized_score(reward_mean)

        # Record the rewards to log.
        model.offline_rewards.append(reward_mean)
        model.offline_rewards_std.append(reward_std)
        # model.offline_normalized_rewards.append(normalized_reward_mean * 100)

        model._dump_logs()
        model.save(f"/workspace/delilog/model/{env_name}/{algo_name}-seed{args.seed}.zip")