import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import gym
import gym.wrappers
import chainerrl.wrappers
from gym.envs.registration import register

from envs.easy_lunar_wrapper import EasyLunar
register(
    id='MultiLunar-v0',
    entry_point='multi_lunar:LunarLander',
    kwargs={'start_indexes': (2,5,8), 'goal_indexes': (2,5,8)},
    max_episode_steps=1000,
    reward_threshold=200,
)

args = None

def set_args(args_to_set):
    global args
    args = args_to_set

def make(test, process_idx=0):
    if args is None:
        print("set args")
    try:
        process_seed = process_idx + args.seed * args.num_envs
    except AttributeError:
        process_seed = args.seed

    env = gym.make(args.env)
    if args.env == 'MultiLunar-v0':
        env = EasyLunar(env)
    
    env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
    env.seed(env_seed)
    # Cast observations to float32 because our model uses float32
    env = chainerrl.wrappers.CastObservationToFloat32(env)
    if args.monitor:
        env = gym.wrappers.Monitor(env, args.outdir)
        pass
    if not test:
        # Scale rewards (and thus returns) to a reasonable range so that
        # training is easier
        env = chainerrl.wrappers.ScaleReward(env, args.reward_scale_factor)
        pass
    if process_idx == 0 and (
            ((args.render_eval and test) or
            (args.render_train and not test)) or args.render):
        env = chainerrl.wrappers.Render(env)
        pass
    return env
