from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import logging
import statistics
import time

import numpy as np

import pfrl
from pfrl.experiments.evaluator import Evaluator
from pfrl.experiments.evaluator import record_stats, save_agent

#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

test_ibe = True

if test_ibe:
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    #ax1 = fig.add_subplot(111, projection='3d')
    #ax2 = fig.add_subplot(122, projection='3d')


def plot_ibe(diff, intps, xp):
    LEFT = False
    diff = xp.concatenate(diff)
    dx = diff[:, 0].get()
    dvx = diff[:, 2].get()
    intps = xp.concatenate(intps)
    intp_to_right = intps[:, 0, 1].get()
    if LEFT:
        intp_to_left = intps[:, 1, 1].get()
    clock = str(time.clock())

    ax1.cla()
    ax2.cla()
    ax1.scatter(dx, intp_to_right, marker='.', c='black')
    ax1.set_title('To right')
    if LEFT:
        ax2.scatter(dx, intp_to_left, marker='.', c='black')
    ax2.set_title('To left')
    ax1.set_ylim(0,1)
    ax2.set_ylim(0,1)
    plt.savefig('figs/'+clock+'dx.png')
   
    ax1.cla()
    ax2.cla()
    ax1.scatter(dvx, intp_to_right, marker='.', c='black')
    ax1.set_title('To right')
    if LEFT:
        ax2.scatter(dvx, intp_to_left, marker='.', c='black')
    ax2.set_title('To left')
    ax1.set_ylim(0,1)
    ax2.set_ylim(0,1)
    plt.savefig('figs/'+clock+'dvx.png')

    if False:
        ax1.cla()
        ax2.cla()
        ax1.scatter(dy, intp_to_right, marker='.', c='black', label='to right')
        ax2.scatter(dy, intp_to_left, marker='.', c='black', label='to left')
        plt.legend()
        plt.savefig('figs/'+clock+'dy.png')

        ax1.cla()
        ax2.cla()
        ax1.scatter(dvy, intp_to_right, marker='.', c='black', label='to right')
        ax2.scatter(dvy, intp_to_left, marker='.', c='black', label='to left')
        plt.legend()
        plt.savefig('figs/'+clock+'dvy.png')

def plot_ibe2(agent, xp):
    dx = xp.arange(-1, 1, 0.01, dtype='f').reshape((-1,1))
    intps = agent.ibe_model.interpret(dx).all_prob.array
    dx = dx.get()
    intp_to_right = intps[:, 0, 1].get()
    intp_to_left = intps[:, 1, 1].get()
    ax1.cla()
    ax2.cla()
    ax1.scatter(dx, intp_to_right, marker='.', c='black', label='to right')
    ax2.scatter(dx, intp_to_left, marker='.', c='black', label='to left')
    ax1.set_ylim(0,1)
    ax2.set_ylim(0,1)
    plt.savefig('fig/'+str(time.clock())+'.png')



def batch_run_evaluation_episodes(
    env,
    agent,
    n_steps,
    n_episodes,
    max_episode_len=None,
    logger=None,
):
    """Run multiple evaluation episodes and return returns in a batch manner.

    Args:
        env (VectorEnv): Environment used for evaluation.
        agent (Agent): Agent to evaluate.
        n_steps (int): Number of total timesteps to evaluate the agent.
        n_episodes (int): Number of evaluation runs.
        max_episode_len (int or None): If specified, episodes
            longer than this value will be truncated.
        logger (Logger or None): If specified, the given Logger
            object will be used for logging results. If not
            specified, the default logger of this module will
            be used.

    Returns:
        List of returns of evaluation runs.
    """
    assert (n_steps is None) != (n_episodes is None)

    logger = logger or logging.getLogger(__name__)
    num_envs = env.num_envs
    episode_returns = dict()
    episode_lengths = dict()
    episode_indices = np.zeros(num_envs, dtype='i')
    episode_idx = 0
    for i in range(num_envs):
        episode_indices[i] = episode_idx
        episode_idx += 1
    episode_r = np.zeros(num_envs, dtype=np.float64)
    episode_len = np.zeros(num_envs, dtype='i')

    obss = env.reset()
    rs = np.zeros(num_envs, dtype='f')

    termination_conditions = False
    timestep = 0

    ibe_diffs = []
    ibe_intps = []

    while True:
        # a_t
        actions = agent.batch_act(obss)
        timestep += 1
        # o_{t+1}, r_{t+1}
        obss, rs, dones, infos = env.step(actions)
        episode_r += rs
        episode_len += 1
        # Compute mask for done and reset
        if max_episode_len is None:
            resets = np.zeros(num_envs, dtype=bool)
        else:
            resets = (episode_len == max_episode_len)
        resets = np.logical_or(
            resets, [info.get('needs_reset', False) for info in infos])

        # Make mask. 0 if done/reset, 1 if pass
        end = np.logical_or(resets, dones)
        not_end = np.logical_not(end)

        for index in range(len(end)):
            if end[index]:
                episode_returns[episode_indices[index]] = episode_r[index]
                episode_lengths[episode_indices[index]] = episode_len[index]
                # Give the new episode an a new episode index
                episode_indices[index] = episode_idx
                episode_idx += 1

        episode_r[end] = 0
        episode_len[end] = 0

        # find first unfinished episode
        first_unfinished_episode = 0
        while first_unfinished_episode in episode_returns:
            first_unfinished_episode += 1

        # Check for termination conditions
        eval_episode_returns = []
        eval_episode_lens = []
        if n_steps is not None:
            total_time = 0
            for index in range(first_unfinished_episode):
                total_time += episode_lengths[index]
                # If you will run over allocated steps, quit
                if total_time > n_steps:
                    break
                else:
                    eval_episode_returns.append(episode_returns[index])
                    eval_episode_lens.append(episode_lengths[index])
            termination_conditions = total_time >= n_steps
            if not termination_conditions:
                unfinished_index = np.where(
                    episode_indices == first_unfinished_episode)[0]
                if total_time + episode_len[unfinished_index] >= n_steps:
                    termination_conditions = True
                    if first_unfinished_episode == 0:
                        eval_episode_returns.append(
                            episode_r[unfinished_index])
                        eval_episode_lens.append(
                            episode_len[unfinished_index])

        else:
            termination_conditions = first_unfinished_episode >= n_episodes
            if termination_conditions:
                # Get the first n completed episodes
                for index in range(n_episodes):
                    eval_episode_returns.append(episode_returns[index])
                    eval_episode_lens.append(episode_lengths[index])

        if termination_conditions:
            # If this is the last step, make sure the agent observes reset=True
            resets.fill(True)

        # Agent observes the consequences.

        if termination_conditions:
            break
        else:
            obss = env.reset(not_end)

    for i, (epi_len, epi_ret) in enumerate(
            zip(eval_episode_lens, eval_episode_returns)):
        logger.info('evaluation episode %s length: %s R: %s',
                    i, epi_len, epi_ret)
    return [float(r) for r in eval_episode_returns]


def eval_performance(env, agent, n_steps, n_episodes, max_episode_len=None,
                     logger=None):
    """Run multiple evaluation episodes and return statistics.

    Args:
        env (Environment): Environment used for evaluation
        agent (Agent): Agent to evaluate.
        n_steps (int): Number of timesteps to evaluate for.
        n_episodes (int): Number of evaluation episodes.
        max_episode_len (int or None): If specified, episodes longer than this
            value will be truncated.
        logger (Logger or None): If specified, the given Logger object will be
            used for logging results. If not specified, the default logger of
            this module will be used.
    Returns:
        Dict of statistics.
    """

    assert (n_steps is None) != (n_episodes is None)

    if isinstance(env, pfrl.env.VectorEnv):
        scores = batch_run_evaluation_episodes(
            env, agent, n_steps, n_episodes,
            max_episode_len=max_episode_len,
            logger=logger)
    else:
        logger.fatal("non-batch not implemented")
        import sys
        sys.exit(1)
    stats = dict(
        episodes=len(scores),
        mean=statistics.mean(scores),
        median=statistics.median(scores),
        stdev=statistics.stdev(scores) if len(scores) >= 2 else 0.0,
        max=np.max(scores),
        min=np.min(scores))
    return stats




class MyEvaluator(Evaluator):
    """Object that is responsible for evaluating a given agent.

    Args:
        agent (Agent): Agent to evaluate.
        env (Env): Env to evaluate the agent on.
        n_steps (int): Number of timesteps used in each evaluation.
        n_episodes (int): Number of episodes used in each evaluation.
        eval_interval (int): Interval of evaluations in steps.
        outdir (str): Path to a directory to save things.
        max_episode_len (int): Maximum length of episodes used in evaluations.
        step_offset (int): Offset of steps used to schedule evaluations.
        save_best_so_far_agent (bool): If set to True, after each evaluation,
            if the score (= mean of returns in evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
    """

    def __init__(self,
                 agent,
                 env,
                 n_steps,
                 n_episodes,
                 eval_interval,
                 outdir,
                 max_episode_len=None,
                 step_offset=0,
                 save_best_so_far_agent=True,
                 logger=None,
                 ):
        super().__init__(agent, env, n_steps, n_episodes, eval_interval,
                outdir, max_episode_len, step_offset, save_best_so_far_agent,
                logger)

    def evaluate_and_update_max_score(self, t, episodes):
        eval_stats = eval_performance(
            self.env, self.agent, self.n_steps, self.n_episodes,
            max_episode_len=self.max_episode_len,
            logger=self.logger)
        elapsed = time.time() - self.start_time
        custom_values = tuple(tup[1] for tup in self.agent.get_statistics())
        mean = eval_stats['mean']
        values = (t,
                  episodes,
                  elapsed,
                  mean,
                  eval_stats['median'],
                  eval_stats['stdev'],
                  eval_stats['max'],
                  eval_stats['min']) + custom_values
        record_stats(self.outdir, values)
        if mean > self.max_score:
            self.logger.info('The best score is updated %s -> %s',
                             self.max_score, mean)
            self.max_score = mean
            if self.save_best_so_far_agent:
                save_agent(self.agent, "best", self.outdir, self.logger)
        else:
            self.logger.info('Score: %s', mean)
        save_agent(self.agent, str(t), self.outdir, self.logger)
        return mean

