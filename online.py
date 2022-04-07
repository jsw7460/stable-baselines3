import argparse

import gym

from stable_baselines3 import *
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_qs", type=int, default=2)
    parser.add_argument("--collect_size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--perturb", type=float, default=0.0)

    args = parser.parse_args()
    env = gym.make(f"{args.env_name}")
    env_name = env.unwrapped.spec.id        # String. Used for save the model file.

    file_name = f"{env_name}-n_qs{args.n_qs}-seed{args.seed}"

    if args.collect_size > 0:        # Collect expert data using expert policy
        model = SAC.load(f"/workspace/expertdata/{env_name}/best_model")

        collect_size = args.collect_size
        SAC.collect_expert_traj(
            model=model,
            env=env,
            save_data_path=f"/workspace/expertdata/{env_name}/expert_buffer-{collect_size}-perturb{args.perturb}",
            collect_size=collect_size,
            deterministic=True,
            perturb=args.perturb
        )
        exit()

    policy_kwargs = {"n_critics": args.n_qs}
    model = SAC(
        "MlpPolicy",
        env=env,
        verbose=1,
        seed=args.seed,
        policy_kwargs=policy_kwargs,
    )

    algo_name = model.__class__.__name__.split(".")[-1]
    callback = EvalCallback(
        model.env,
        eval_freq=5000,
        # best_model_save_path=f"/workspace/expertdata/{env_name}"
    )
    model.learn(1000000, callback=callback)
    # for i in range(10000000):
    #     model.learn(500, reset_num_timesteps=False)
    #     model.save(f"/workspace/expertdata/{env_name}/best_model")