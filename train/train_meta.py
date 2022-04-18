import argparse
import logging
import numpy as np
from pathlib import Path
import json
import os
import sys
import random
ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)
sys.path.append(str(ROOT_DIR))


import torch
from torch import nn
import torch.nn.functional as F

import utils.record
import agents
from agents import train_params
import envs.wrappers
from envs import goal_xs, map_goal_index
from utils.datasets import Dataset

logging.basicConfig(level=logging.INFO)

BATCH_SIZE = 128
NUM_EPOCHS = 20
NUM_WORKERS = 8

SAVE_PATH = Path('data/meta_models/teacher/')

label_mode = 'g0'
score_mode = 'softmax'



class Human():
    CLASSES = {label: value for label, value in zip(('left', 'stay', 'right'), range(3))}
    CLASSES_INVERSED = {value: label for label, value in CLASSES.items()}

    def __init__(self, mode, **kargs):
        assert mode in ('true_belief', 'random', 'fixed', 'bot') + tuple(goal_xs.keys())
        # bot: indefinite output
        self.mode = mode
        # goal that human believes the agent has (level-1 inference)
        if mode == 'fixed':
            self.g1 = kargs['g1']
        elif mode == 'random':
            self.update_belief()

    def update_belief(self, index=None):
        if self.mode == 'random' and index is None:
            self.g1 = np.random.choice(list(goal_xs.keys()))
        elif index is not None:
            self.g1 = list(goal_xs.keys())[index]

    def believe(self, state, mode=None):
        if mode is None:
            mode = self.mode
        if self.mode == 'bot':
            g = 1
        elif mode == 'true_belief':
            for key, value in goal_xs.items():
                if value == state[8]:
                    g = map_goal_index[key]
                    break
            else:
                raise KeyError
        else:
            g = map_goal_index[self.g1]
        return g

    def utter(self, state, mode=None):
        if mode is None:
            mode = self.mode
        if self.mode == 'bot':
            u = self.CLASSES['stay']
        elif mode == 'true_belief':
            u = self._utter_with_belief(state)
        else:
            u = self._utter_with_belief(state, goal_xs[self.g1])
        return u

    def _utter_with_belief(self, state, g_x=None):
        x = state[0]
        #print(g_x, state[8])
        if g_x is None:
            # Human knows agent's actual goal.
            g_x = state[8]

        if x - g_x > 0.2:
            return self.CLASSES['left']
        elif x - g_x < -0.2:
            return self.CLASSES['right']
        else:
            return self.CLASSES['stay']

def calc_accuracy(y, t, mask=None):
    # t.shape = (batch_size) / y.shape = (batch_size, n_features).
    max_vals, max_indices = torch.max(y, 1)
    if mask is not None:
        print(y.shape, mask.shape)
        max_indices = max_indices[~mask]
        t = t[~mask]
    batch_size = len(t)
    acc = (max_indices == t).sum().data / batch_size
    acc = acc.to('cpu').item()
    return acc
