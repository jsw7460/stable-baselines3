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

    normalizing = None
    if args.env_name == "halfcheetah-medium-v2":
        normalizing = 29.01712417602539
    elif args.env_name == "halfcheetah-expert-v2":
        normalizing = 32.98865509033203
    elif args.env_name == "hopper-medium-v2":
        normalizing = 10.0
    elif args.env_name == "hopper-expert-v2":
        normalizing = 10.0
    elif args.env_name == "walker2d-medium-v2":
        normalizing = 10.0
    elif args.env_name == "walker2d-expert-v2":
        normalizing = 10.0

    env = gym.make(args.env_name)
    env_name = env.unwrapped.spec.id  # String. Used for save the model file.

    expert_data_path = f"/workspace/expertdata/dttrajectory/{args.env_name}"

    filename_head = f"/workspace/delilog/"
    filename_tail = f"{env_name}/" \
                    f"{args.algo}" \
                    f"-context{args.context_length}" \
                    f"-grad{int(args.grad_flow)}" \
                    f"-seed{args.seed}"

    model_kwargs = {}
    algo = None
    if args.algo == "delig":
        model = DeliG.load_deli(filename_head + "model/" + filename_tail, env=env, print_system_info=True)
    elif args.algo == "delic":
        model = DeliC.load_deli(filename_head + "model/" + filename_tail, env=env)
    elif args.algo == "delimg":
        model = DeliMG.load_deli(filename_head + "model/" + filename_tail, env=env)

    evaluator = functools.partial(evaluate_deli, context_length=args.context_length, normalizing_factor=normalizing)

    reward, *_ = evaluator(model, env, n_eval_episodes=10)
    print("******************************")
    print("\t ENV:", args.env_name)
    print("\t Rewards mean", reward)
    print("\t Normalized reward", env.get_normalized_score(reward) * 100)
    print("******************************")