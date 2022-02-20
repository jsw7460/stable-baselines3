import argparse
import copy

import gym
import torch as th

from stable_baselines3 import TQCBC, TQCBEAR
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.tqc.tqc_eval import evaluate_tqc_policy
from datetime import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--env_name", type=str, default="halfcheetah")
    parser.add_argument("--degree", type=str, default="medium")

    parser.add_argument("--algo", type=str)
    parser.add_argument("--date", type=str)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--total_timestep", type=int, default=1_000_000)

    parser.add_argument("--n_qs", type=int, default=2)
    parser.add_argument("--grad_step", type=int, default=1)

    parser.add_argument("--use_gumbel", action="store_true")
    parser.add_argument("--temper", type=float, default=0.5)

    parser.add_argument("--n_quantiles", type=int, default=51, help="Only for TQC")
    parser.add_argument("--drop", type=int, default=23)
    parser.add_argument("--use_uncertainty", action="store_true")
    parser.add_argument("--uc_type", choices=["aleatoric", "epistemic", "both"], default="epistemic")
    parser.add_argument("--use_trunc", action="store_true")
    parser.add_argument("--min_clip", type=int, default=51)
    parser.add_argument("--policy", choices=["bc", "bear"], default="bear")
    parser.add_argument("--mmd_thresh", type=float, default=0.05)
    parser.add_argument("--pp", action="store_true", help="Use policy penalty")

    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_reg", type=float, default=0.0, help="Use uncertainty regularization on evaluation time")
    parser.add_argument("--warmup", type=int, default=0)

    args = parser.parse_args()

    env = gym.make(f'{args.env_name}-{args.degree}-v2')
    env_name = env.unwrapped.spec.id        # String. Used for save the model file.

    # Tensorboard file name.
    month = str(datetime.today().month)
    month = "0" + month if month[0] != "0" and len(month) == 1 else month
    day = datetime.today().day
    # If there is a specified date, then use it
    _date = args.date if args.date is not None else f"{month}-{day}"

    board_file_name = f"{_date}/" \
                      f"{env_name}" \
                      f"-n_qs{args.n_qs}" \
                      f"-{args.uc_type}" \
                      f"-n_quan{args.n_quantiles}" \
                      f"-drop{args.drop}" \
                      f"-clip{args.min_clip}" \
                      f"-uncertainty{int(args.use_uncertainty)}" \
                      f"-mmd" \
                      f"-seed{args.seed}"

    if args.use_trunc:
        board_file_name += "-trunc"
    if args.pp:
        board_file_name += "-pp"

    policy_kwargs = {"n_critics": args.n_qs, "activation_fn": th.nn.ReLU, "n_quantiles": args.n_quantiles}

    if args.policy == "bc":
        algo = TQCBC
    elif args.policy == "bear":
        algo = TQCBEAR
    else:
        raise NotImplementedError

    # In debugging, do not save tensorboard.
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
        "min_clip": args.min_clip,
    }

    if args.policy == "bear":
        algo_config.update({
            "uc_type": args.uc_type,
            "use_uncertainty": args.use_uncertainty,
            "top_quantiles_to_drop_per_net": args.drop,
            "warmup_step": args.warmup,
            "truncation": args.use_trunc,
            "mmd_thresh": args.mmd_thresh,
            "use_policy_penalty": args.pp
        })

    model = algo(**algo_config)

    algo_name = model.__class__.__name__.split(".")[-1]
    file_name = algo_name + "-" + board_file_name   # Model parameter file name.

    evaluation_model = copy.deepcopy(model)
    evaluation_fn = evaluate_tqc_policy if args.eval_reg > 0 else evaluate_policy
    eval_config = {
        "model": None,
        "env": None,
        "n_eval_episodes": 10,
    }

    if args.eval:
        print("Evaluation Mode\n")
        print(f"FILE: {file_name}")
        evaluation_model = algo.load(f"../GQEdata/results/{file_name}", env=model.env, device="cpu")
        eval_config.update({"model": evaluation_model, "env": model.env, "unc_coef": args.eval_reg})

        print("Model Load!")
        reward_mean, reward_std = evaluation_fn(**eval_config)
        print("\tREWARD MEAN:", reward_mean)
        print("\tNORMALIZED REWARD MEAN:", env.get_normalized_score(reward_mean) * 100)
        print("\tREWARD STD:", reward_std)
        exit()

    for i in range(args.total_timestep // args.log_interval):
        # Train the model
        model.learn(args.log_interval, reset_num_timesteps=False,)

        # Evaluate the model. By creating a separated model, avoid the interaction with environments of training model.
        evaluation_model.set_parameters(model.get_parameters())
        eval_config.update({"model": evaluation_model, "env": model.env})
        reward_mean, reward_std = evaluation_fn(**eval_config)
        normalized_reward_mean = env.get_normalized_score(reward_mean)

        # Record the rewards to log.
        model.offline_rewards.append(reward_mean)
        model.offline_rewards_std.append(reward_std)
        model.offline_normalized_rewards.append(normalized_reward_mean * 100)

        # Logging
        model.logger.record("config/eval_fn", evaluation_fn.__name__)
        model.logger.record("config/eval_reg", "True" if args.eval_reg > 0 else "False")
        model.dump_logs()
        if not args.debug:
            print(f"Model save to ../GQEdata/results/{file_name}")
            model.save(f"../GQEdata/results/{file_name}")
