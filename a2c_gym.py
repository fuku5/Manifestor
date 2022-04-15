import argparse
import functools
import logging

import numpy as np
import torch
from torch import nn

import gym

import pfrl
from pfrl.agents import a2c
from pfrl.wrappers import atari_wrappers

import envs.wrappers
from agents import a2c as my_a2c

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MultiLunar-v0")
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 31)")
    parser.add_argument("--outdir", type=str, default="data/a2c")
    parser.add_argument("--steps", type=int, default=8 * 10 ** 7)
    parser.add_argument("--update-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--gamma", type=float, default=0.995, help="discount factor")
    parser.add_argument("--rmsprop-epsilon", type=float, default=1e-5)
    parser.add_argument(
        "--use-gae",
        action="store_true",
        default=False,
        help="use generalized advantage estimation",
    )
    parser.add_argument("--tau", type=float, default=0.95, help="gae parameter")
    parser.add_argument(
        "--alpha", type=float, default=0.99, help="RMSprop optimizer alpha"
    )
    parser.add_argument("--eval-interval", type=int, default=10 ** 5)
    parser.add_argument("--eval-n-runs", type=int, default=24)
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--n-hidden-size", type=int, default=2048)
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0, help="value loss coefficient"
    )
    parser.add_argument(
        "--gpu",
        "-g",
        type=int,
        default=4,
        help="GPU ID (negative value indicates CPU)",
    )
    parser.add_argument("--num-envs", type=int, default=12)
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help=(
            "Monitor env. Videos and additional information are saved as output files."
        ),
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
    )
    parser.add_argument('--reward-scale-factor', type=float, default=0.01)

    # non-training parms
    parser.add_argument('--record',type=str, default=None) 
    parser.add_argument('--start_indexes',type=str, default='5') 
    parser.add_argument('--end_indexes',type=str, default='2,5,8') 
    parser.add_argument('--flatten', action='store_true')
    parser.add_argument('--will_change_goal', action='store_true')

    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    # If you use more than one processes, the results will be no longer
    # deterministic even with the same random seed.
    pfrl.utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 31

    args.outdir = pfrl.experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    if args.record:
        import utils.record
        utils.record.init()
        utils.record.out_name = args.record

    gym.envs.registration.register(
        id='MultiLunar-v0',
        entry_point='multi_lunar:LunarLander',
        kwargs={
            'start_indexes': tuple(map(int, args.start_indexes.split(','))), 
            'goal_indexes': tuple(map(int, args.end_indexes.split(','))),
            'flatten': args.flatten,
            'noflag': True,
            'ground_marker': False,
            'no_particles': True,
            'will_change_goal': args.will_change_goal
            },
        max_episode_steps=1000,
        reward_threshold=200,
    )

    def make_env(process_idx, test):
        # Use different random seeds for train and test envs
        process_seed = process_seeds[process_idx]
        env_seed = 2 ** 31 - 1 - process_seed if test else process_seed
        env = gym.make(args.env)
        env = envs.wrappers.EasyLunar(env)
        env = pfrl.wrappers.CastObservationToFloat32(env)
        env = pfrl.wrappers.ScaleReward(env, args.reward_scale_factor)
        if args.record:
            env = envs.wrappers.Record(env, save_img=args.render)
        env.seed(int(env_seed))
        if args.monitor:
            env = pfrl.wrappers.Monitor(
                env, args.outdir, mode="evaluation" if test else "training"
            )
        if args.render:
            env = pfrl.wrappers.Render(env)
        return env

    def make_batch_env(test):
        return pfrl.envs.MultiprocessVectorEnv(
            [
                functools.partial(make_env, idx, test)
                for idx, env in enumerate(range(args.num_envs))
            ]
        )

    sample_env = make_env(0, test=False)
    obs_channel_size = sample_env.observation_space.low.shape[0]
    n_actions = sample_env.action_space.n

    model = my_a2c.make_A2C_model(obs_channel_size, args.n_hidden_size, n_actions) 
    optimizer = pfrl.optimizers.RMSpropEpsInsideSqrt(
        model.parameters(),
        lr=args.lr,
        eps=args.rmsprop_epsilon,
        alpha=args.alpha,
    )

    agent = a2c.A2C(
        model,
        optimizer,
        gamma=args.gamma,
        gpu=args.gpu,
        num_processes=args.num_envs,
        update_steps=args.update_steps,
        phi=lambda x: x,
        use_gae=args.use_gae,
        tau=args.tau,
        max_grad_norm=args.max_grad_norm,
    )

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = pfrl.experiments.eval_performance(
            env=make_batch_env(test=True),
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
        )
        print(
            "n_runs: {} mean: {} median: {} stdev: {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:
        import experiments
        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(test=False),
            eval_env=make_batch_env(test=True),
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=True,
            log_interval=1000,
        )


if __name__ == "__main__":
    main()
