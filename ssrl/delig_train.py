import argparse

import torch as th
import gym

from models import DeliG, DeliG3, DeliG4, DeliG5
from models.deli import evaluate_delig, evaluate_delig3, evaluate_delig4

import functools

REMOVE_DIM = [3, 4, 5, 6]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env_name", type=str, default="MountainCarContinuous-v0")
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
    parser.add_argument("--learn_latent", action="store_true")

    args = parser.parse_args()

    env = gym.make(args.env_name)
    if len(REMOVE_DIM) > 0:
        import sys
        sys.path.append("..")
        from pomdp_envs.pomdp_util import RemoveDim
        env = RemoveDim(env, REMOVE_DIM)

    env_name = env.unwrapped.spec.id  # String. Used for save the model file.

    expert_data_path = f"/workspace/expertdata/{args.env_name}/expert_buffer-{args.buffer_size}-perturb{args.perturb}"

    model_kwargs = {}
    if args.var == 0:
        algo = DeliG
        evaluator = functools.partial(evaluate_delig, context_length=args.context_length)
        algo_name = "delig0"
    elif args.var == 3:
        algo = DeliG3
        evaluator = functools.partial(evaluate_delig3, context_length=args.context_length)
        algo_name = "delig3"
    elif args.var == 4:
        algo = DeliG4
        evaluator = functools.partial(evaluate_delig4, context_length=args.context_length)
        model_kwargs["learn_latent"] = args.learn_latent
        algo_name = "deli4"
    elif args.var == 5:
        algo = DeliG5
        evaluator = functools.partial(evaluate_delig3, context_length=args.context_length)
        algo_name = "deli5"

    else:
        raise NotImplementedError

    tensorboard_log = f"/workspace/delilog/tensorboard/" \
                      f"{env_name}/" \
                      f"{algo_name}" \
                      f"-buffer{args.buffer_size}" \
                      f"-perturb{args.perturb}" \
                      f"-context{args.context_length}" \
                      f"-remove{REMOVE_DIM}" \
                      f"-seed{args.seed}"

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
        "pomdp_remove_dim": REMOVE_DIM
    }

    model = algo(**model_kwargs)
    model.learn(1, reset_num_timesteps=False)

    random_model = algo(**model_kwargs)
    random_reward, *_ = evaluator(random_model, env, n_eval_episodes=1)

    for i in range(1000000):
        model.learn(5000, reset_num_timesteps=False)
        model.set_training_mode(False)
        reward_mean, *_ = evaluator(model, env, n_eval_episodes=3)
        model.logger.record("performance/reward/mean", reward_mean)
        model.logger.record("performance/reward/random", random_reward)
        model._dump_logs()
        model.save(
            f"/workspace/delilog/model/{env_name}/deli-dropout{args.dropout}"
            f"-seed{args.seed}"
            f"-buffer{args.buffer_size}"
            f"-perturb{args.perturb}.zip"
        )
        model.set_training_mode(True)
