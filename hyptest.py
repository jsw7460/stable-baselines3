import argparse
import copy
from datetime import datetime

import gym
import torch as th

from stable_baselines3 import TD3, SAC, CQL, MIN, BCQ, BEAR
from stable_baselines3.common.evaluation import evaluate_policy


def get_algorithm(name: str):
    if name == "td3" or name == "TD3":
        return TD3
    elif name == "sac" or name == "SAC":
        return SAC
    elif name == "cql" or name == "CQL":
        return CQL
    elif name == "min" or name == "MIN":
        return MIN
    elif name == "bcq" or name == "BCQ":
        return BCQ
    elif name == "bear" or name == "BEAR":
        return BEAR
    else:
        raise NotImplementedError("No algorithm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="halfcheetah")
    parser.add_argument("--degree", type=str, default="random")

    parser.add_argument("--algo", type=str)
    parser.add_argument("--date", type=str)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--total_timestep", type=int, default=1_000_000)

    parser.add_argument("--n_qs", type=int, default=2)
    parser.add_argument("--grad_step", type=int, default=1)

    parser.add_argument("--use_gumbel", action="store_true")
    parser.add_argument("--temper", type=float, default=0.5)

    parser.add_argument("--c", type=float, default=10.0)            # Halfcheetah: 10.0, Else: 1.0
    parser.add_argument("--thresh", type=float, default=10.0)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    env = gym.make(f'{args.env_name}-{args.degree}-v2')
    env_name = env.unwrapped.spec.id        # String. Used for save the model file.

    # Tensorboard file name.
    month = str(datetime.today().month)
    month = "0" + month if month[0] != "0" and len(month) == 1 else month
    day = datetime.today().day
    # If there is a specified date, then use it
    _date = args.date if args.date is not None else f"{month}-{day}"

    board_file_name = f"hyptest/{_date}/" \
                      f"{env_name}" \
                      f"-n_qs{args.n_qs}" \
                      f"-seed{args.seed}"

    algo = get_algorithm(args.algo)
    policy_kwargs = {"n_critics": args.n_qs, "activation_fn": th.nn.ReLU}

    if algo == CQL:
        board_file_name += f"-c{args.c}" \
                           f"-thresh{args.thresh}"

    tensorboard_log_name = f"../GQEdata/board/{board_file_name}" if not args.debug else None

    algo_config = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_starts": 0,
        "verbose": 1,
        "policy_kwargs": policy_kwargs,
        "seed": args.seed,
        "without_exploration": True,
        "gumbel_ensemble": args.use_gumbel,
        "gumbel_temperature": args.temper,
        "tensorboard_log": tensorboard_log_name,
        "gradient_steps": args.grad_step,
        "device": args.device,
    }

    if algo == CQL:
        algo_config["conservative_weight"] = args.c
        algo_config.update({
            "conservative_weight": args.c,
            "lagrange_thresh": args.thresh,
        })

    model = algo(
        **algo_config
    )

    algo_name = model.__class__.__name__.split(".")[-1]
    file_name = algo_name + "-" + board_file_name   # Model parameter file name.

    evaluation_model = copy.deepcopy(model)
    if args.eval:
        print("Evaluation Mode\n")
        evaluation_model = algo.load(f"../GQEdata/results/{file_name}", device="cpu")
        print("Model Load!")
        reward_mean, reward_std = evaluate_policy(evaluation_model, model.env)
        print("\tREWARD MEAN:", reward_mean)
        print("\tNORMALIZED REWARD MEAN:", env.get_normalized_score(reward_mean) * 100)
        print("\tREWARD STD:", reward_std)
        exit()

    for i in range(args.total_timestep // args.log_interval):
        # Train the model
        model.learn(args.log_interval, reset_num_timesteps=False,)

        # Evaluate the model. By creating a separated model, avoid the interaction with environments of training model.
        evaluation_model.set_parameters(model.get_parameters())
        reward_mean, reward_std = evaluate_policy(evaluation_model, model.env)
        normalized_reward_mean = env.get_normalized_score(reward_mean)

        # Record the rewards to log.
        model.offline_rewards.append(reward_mean)
        model.offline_rewards_std.append(reward_std)
        model.offline_normalized_rewards.append(normalized_reward_mean * 100)

        # Logging
        model._dump_logs()
        model.save(f"../GQEdata/results/{file_name}")
