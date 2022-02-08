import os

from stable_baselines3.a2c import A2C
from stable_baselines3.common.utils import get_system_info
from stable_baselines3.ddpg import DDPG
from stable_baselines3.dqn import DQN
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC
from stable_baselines3.td3 import TD3


from stable_baselines3.c51 import C51
from stable_baselines3.qrdqn import QRDQN
from stable_baselines3.iqn import IQN
from stable_baselines3.cql import CQL
from stable_baselines3.edac import EDAC
from stable_baselines3.wbcq import WBCQ
from stable_baselines3.trpo import TRPO
from stable_baselines3.sepac import SEPAC

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file, "r") as file_handler:
    __version__ = file_handler.read().strip()


def HER(*args, **kwargs):
    raise ImportError(
        "Since Stable Baselines 2.1.0, `HER` is now a replay buffer class `HerReplayBuffer`.\n "
        "Please check the documentation for more information: https://stable-baselines3.readthedocs.io/"
    )
