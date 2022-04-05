from evaluate_policy import evaluate_policy
import argparse

import gym

from ssrl.models.deli.deli import Deli

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--vae_feature_dim", type=int, default=10)
    parser.add_argument("--latent_dim", type=int, default=25)
    parser.add_argument("--additional_dim", type=int, default=1)

    args = parser.parse_args()

    env = gym.make(args.env)
    env_name = env.unwrapped.spec.id  # String. Used for save the model file.

    model = Deli.load_deli(
        f"/workspace/delilog/{env_name}/deli/model/dropout{args.dropout}-seed{args.seed}",
        env=env,
        device="cpu"
    )
    # for param in model.policy.parameters():
    #     param.data = th.randn_like(param)
    rewards, lengths = evaluate_policy(model, env, n_eval_episodes=1)
    print("RUN~!", rewards, lengths)