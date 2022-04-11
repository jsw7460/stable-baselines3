import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback


class HopperMass(gym.Env):
    def __init__(self, dyn=5, unc=2, episode_len=1000, use_gravity: bool = False):
        # Use gravity (or partially observable factor) for expert training.
        self.use_gravity = use_gravity
        self.dynamics = dyn
        self.uncertainty = unc
        self.gravity_theta = 0
        self.env = gym.make('Hopper-v3')
        self.episode_len = episode_len
        self.timesteps = 0

        self.gravity = self.env.sim.model.opt.wind[2]
        self.action_space = self.env.action_space
        if use_gravity:
            pomdp_dim = self.env.observation_space.shape[0] + 1
            self.observation_space = gym.spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(pomdp_dim, )
            )
        else:
            self.observation_space = self.env.observation_space

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if self.use_gravity:
            state = np.hstack([state, self.gravity.copy()])
        self.timesteps += 1
        if self.timesteps == self.episode_len:
            done = True
        if np.random.uniform(0, 1) < 0.1:
            self.mass_step()
        return state, reward, done, info

    def reset(self):
        self.mass_step()
        self.timesteps = 0
        observation = self.env.reset()
        if self.use_gravity:
            observation = np.hstack([observation, self.gravity])
        return observation

    def mass_step(self):
        self.gravity_theta = 2 * np.pi * np.random.normal()
        self.gravity = -9.81 + np.sin(self.gravity_theta) * 100
        self.env.sim.model.opt.gravity[:] = np.array([0.0, 0.0, self.gravity])
        self.env.sim.model.opt.wind[:] = np.random.normal(3) * 9999999
        self.env.sim.set_constants()
        return

    def render(self, mode='human'):
        self.env.render()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gravity", action="store_true")
    parser.add_argument("--load", action="store_true")

    parser.add_argument("--collect", type=int, default=0)
    parser.add_argument("--perturb", type=float, default=0.0)
    args = parser.parse_args()

    # env = HopperMass(dyn=5, unc=2, use_gravity=args.use_gravity)

    from pomdp_util import RemoveDim

    env = RemoveDim(gym.make("Hopper-v3"), [])
    env_name = "HopperRem"

    if args.collect:
        env.use_gravity = True
        agent = SAC.load(f"/workspace/expertdata/pomdp/{env_name}/best_model", env)
        collect_size = args.collect
        SAC.collect_expert_traj(
            model=agent,
            env=env,
            save_data_path=f"/workspace/expertdata/{env_name}/expert_buffer-{collect_size}-perturb{args.perturb}",
            collect_size=collect_size,
            deterministic=True,
            perturb=args.perturb,
            pomdp_hidden_dim=1,
        )

    if args.load:
        agent = SAC.load("/workspace/expertdata/pomdp/best_model", env)
    else:
        agent = SAC('MlpPolicy', env, verbose=1, seed=0)

    callback = None
    if args.use_gravity:
        callback = EvalCallback(agent.env, best_model_save_path=f"/workspace/expertdata/{env_name}pomdp")

    agent.learn(total_timesteps=1000000, callback=callback)
    agent.save('test')
