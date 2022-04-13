import argparse
import copy

import gym
import torch as th

from stable_baselines3 import TD3, SAC, CQL, MIN, BCQ, BEAR, SACAUBCQ, UWAC, TQC, SACOdice, SACMIN
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
    elif name == "sacmin" or name == "SACMIN":
        return SACMIN
    elif name == "bcq" or name == "BCQ":
        return BCQ
    elif name == "bear" or name == "BEAR":
        return BEAR
    elif name == "sacaubcq" or name == "SACAUBCQ":
        return SACAUBCQ
    elif name == "uwac" or name == "UWAC":
        return UWAC
    elif name == "tqc" or name == "TQC":
        return TQC
    elif name == "sacodice" or name == "SACODICE":
        return SACOdice
    else:
        raise NotImplementedError("No algorithm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="halfcheetah")
    parser.add_argument("--degree", type=str, default="medium")

    parser.add_argument("--algo", type=str, default="sac")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--total_timestep", type=int, default=1_000_000)

    parser.add_argument("--n_qs", type=int, default=2)
    parser.add_argument("--grad_step", type=int, default=1)

    parser.add_argument("--use_gumbel", action="store_true")
    parser.add_argument("--temper", type=float, default=0.5)

    parser.add_argument("--eval", action="store_true")

    parser.add_argument("--n_quantiles", type=int, default=16, help="Only for TQC")
    parser.add_argument("--dropout", type=float, default=0.0, help="Only for UWAC")

    args = parser.parse_args()

    env = gym.make(f'{args.env_name}-{args.degree}-v2')
    env_name = env.unwrapped.spec.id        # String. Used for save the model file.

    z = env.get_normalized_score(4800)
    print(z)
    exit()
    # Tensorboard file name.
    board_file_name = f"{env_name}-n_qs{args.n_qs}-gum{args.temper}-seed{args.seed}" if args.use_gumbel \
        else f"{env_name}-n_qs{args.n_qs}-seed{args.seed}"

    algo = get_algorithm(args.algo)
    policy_kwargs = {"n_critics": args.n_qs, "activation_fn": th.nn.ReLU}

    if algo == UWAC:
        policy_kwargs["dropout"] = args.dropout

    model = algo(
        "MlpPolicy",
        env=env,
        learning_starts=0,
        verbose=1,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        without_exploration=True,
        gumbel_ensemble=args.use_gumbel,
        gumbel_temperature=args.temper,
        tensorboard_log=f"../GQEdata/board/{board_file_name}",
        gradient_steps=args.grad_step,
        batch_size=2048,
        ent_coef=0
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
