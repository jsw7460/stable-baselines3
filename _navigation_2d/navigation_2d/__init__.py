from .navigation_env import NavigationEnvDefault, NavigationEnvAcc, NavigationEnvAccLidarObs, NonStationaryNavigation,\
    NavigationEnvAccLidarObsRandomInit, NavigationEnvAccRandomInit, NavigationEnvRandomInit, StationaryNavigation
from gym.envs import register
from .config import *

mode = ['easy', 'normal', 'hard', 'very_hard']

custom_envs = {}
for idx, obs_conf in enumerate(config_set):
    custom_envs['Navi-Vel-Full-Obs-Task{}_{}-v0'.format(idx%8, mode[j//8])] = dict(
                 path='_navigation_2d:NavigationEnvDefault',
                 max_episode_steps=200,
                 kwargs=dict(task_args=obs_conf))
    custom_envs['Navi-Acc-Full-Obs-Task{}_{}-v0'.format(idx%8, mode[j//8])] = dict(
                 path='_navigation_2d:NavigationEnvAcc',
                 max_episode_steps=200,
                 kwargs=dict(task_args=obs_conf))
    custom_envs['Navi-Acc-Lidar-Obs-Task{}_{}-v0'.format(idx%8, mode[j//8])] = dict(
                 path='_navigation_2d:NavigationEnvAccLidarObs',
                 max_episode_steps=200,
                 kwargs=dict(task_args=obs_conf))
    custom_envs['Navi-Vel-Full-Obs-Random-Init-Task{}_{}-v0'.format(idx%8, mode[j//8])] = dict(
                 path='_navigation_2d:NavigationEnvDefaultRandomInit',
                 max_episode_steps=200,
                 kwargs=dict(task_args=obs_conf))
    custom_envs['Navi-Acc-Full-Obs-Random-Init-Task{}_{}-v0'.format(idx%8, mode[j//8])] = dict(
                 path='_navigation_2d:NavigationEnvAccRandomInit',
                 max_episode_steps=200,
                 kwargs=dict(task_args=obs_conf))
    custom_envs['Navi-Acc-Lidar-Obs-Random-Init-Task{}_{}-v0'.format(idx%8, mode[j//8])] = dict(
                 path='_navigation_2d:NavigationEnvAccLidarObsRandomInit',
                 max_episode_steps=200,
                 kwargs=dict(task_args=obs_conf))

for i in range(10):
    for j in range(5):
        custom_envs['Non-Stationary-Navigation_dyn_{}_unc_{}-v0'.format(i, j)] = dict(
                     path='_navigation_2d:NonStationaryNavigation',
                     max_episode_steps=200,
                     kwargs=dict(task_args=config_set_2[5 * i + j]))
for i in range(10):
    for j in range(5):
        custom_envs['Stationary-Navigation_dyn_{}_unc_{}-v0'.format(i, j)] = dict(
                     path='_navigation_2d:StationaryNavigation',
                     max_episode_steps=200,
                     kwargs=dict(task_args=config_set_2[5 * i + j]))


# register each env into
def register_custom_envs():
    for key, value in custom_envs.items():
        arg_dict = dict(id=key,
                        entry_point=value['path'],
                        max_episode_steps=value['max_episode_steps'],
                        kwargs=value['kwargs'])
        if 'reward_threshold' in value.keys():
            arg_dict['reward_threshold'] = value['reward_threshold']
        register(**arg_dict)

register_custom_envs()
