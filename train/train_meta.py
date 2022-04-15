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



def reward_based_loss(y, labels_all0, labels_all1, rewards, goals0, goals1, goals2=None, x=[]):
    # first utterance in 60 frames
    labels0 = labels_all0[:,0]
    labels1 = labels_all1[:,0]
    nll = -F.log_softmax(y, 1)

    scores_orig = [rewards[key].unsqueeze(1) for key in [2, 5, 8]]
    scores_orig = torch.hstack(scores_orig).to(device)
    #print(scores_orig)
    #print(goals0)
    # scores: (BATCH_SIZE, 3)

    logging.debug('label_mode: ' + label_mode)
    if label_mode == 'g0':
        # agent assumes human's utterance is based on the agent's actual goal
        u_mask = nn.functional.one_hot(labels0, 3)
        g_prob = nn.functional.one_hot(goals0, 3)
    elif label_mode == 'g1':
        # agent knows the goal attributed by a human observer
        u_mask = nn.functional.one_hot(labels1, 3)
        g_prob = nn.functional.one_hot(goals1, 3).float()
    elif label_mode == 'g2':
        # based on agent's evel-2 inference of what goal a human attributes
        u_mask = nn.functional.one_hot(labels2, 3)
        g_prob = nn.functional.one_hot(goals2, 3)
    else:
        raise AssertionError
    
    logging.debug('score_mode: ' + score_mode)
    if score_mode == 'softmax':
        scores = F.softmax(scores_orig, 1)
    elif score_mode == 'mean_error':
        scores = 0.1 * (scores_orig - scores_orig.mean(axis=1, keepdims=True)) / scores_orig.mean(axis=1, keepdims=True)
    elif score_mode == 'ranking':
        scores = torch.argsort(torch.argsort(scores_orig, dim=1), dim=1).type(torch.float32) - 1.
    elif score_mode == 'mean_error2':
        ll = torch.log(1-F.softmax(y, 1))
        scores = 0.1 * (scores_orig - scores_orig.mean(axis=1, keepdims=True)) # / scores_orig.mean(axis=1, keepdims=True)
        nll[scores < 0] = ll[scores < 0]
    elif score_mode == 'double_softmax':
        nll_pos = nll
        scores_pos = F.softmax(scores_orig, 1) 
        scores_pos *= g_prob
        loss_pos = (nll_pos * u_mask) * scores_pos.sum(axis=1, keepdims=True)
        
        if True:
            nll_neg = -torch.log(1. - F.softmax(y, 1))
            scores_neg = (1. - F.softmax(scores, 1)) 
            scores_neg *= g_prob
            loss_neg = (nll_neg * u_mask) * scores_neg.sum(axis=1, keepdims=True)
        else:
            loss_neg = 0.
        return torch.mean(loss_pos + loss_neg)
    else:
        raise AssertionError

    #print(scores.sum(axis=0))
    scores *= g_prob
    #print(u_mask.shape, scores.shape)
    scores = scores.sum(axis=1, keepdims=True)
    if False:
        #print(x.shape)
        for line in zip(scores, goals0, goals1, scores_orig, x.transpose(1,0)[:,:,0:4]):
            if True:
                if line[0] > 0.5 and line[1] != line[2]:
                    print("a", line)
                if line[0] < 0.5 and line[1] == line[2]:
                    print("b", line)
            else:
                if line[0] < 0.5:
                    print(line)
    #print(nll*u_mask*scores)
    if False:
        return torch.mean(nll * u_mask * scores)
    else:
        return torch.mean(nll * u_mask)


def train_meta(memory, model_type, human_mode, remove_goal):
    bound = int(len(memory)*0.5)
    #memory_train = sum([memory[i::10] for i in range(0,8)], list())
    #memory_test = sum([memory[i::10] for i in range(8,10)], list())

    params = train_params[model_type]

    human = Human(human_mode)
    train_dataset = Dataset(memory[:bound], human, remove_goal=remove_goal,
            flatten=params['flatten'], info=True)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True)
    test_dataset = Dataset(memory[bound:], human, remove_goal=remove_goal,
            flatten=params['flatten'],  info=True)
    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True)

    print(len(train_dataset), len(test_dataset))
    sample_obs = train_dataset[0][0]

    # main model
    criterion = reward_based_loss
    n_in = params['calc_n_in'](sample_obs)
    model = params['model'](n_in=n_in, **params['kargs'])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    train_model = True

    params['score_mode'] = score_mode
    params['label_mode'] = label_mode

    def run(data_loader, train=True):
        loss_total, loss2_total, acc0, acc1, acc_g0, acc_g1 = [0.] * 6
        i_max = len(data_loader)
        log_interval = i_max // 10

        for i, data in enumerate(data_loader):
            inputs, labels0, labels1, rewards, goals0, goals1 = data
            inputs = params['pre_func'](inputs)
            inputs = inputs.to(device)
            labels0 = labels0.to(device)
            labels1 = labels1.to(device)
            goals0 = goals0.to(device)
            goals1 = goals1.to(device)

            if train:
                if train_model:
                    optimizer.zero_grad()

            y = model(inputs)
            loss = criterion(y, labels0, labels1, rewards, goals0, goals1, None, inputs)

            if train:
                if train_model:
                    loss.backward()
                    optimizer.step()

            loss_total += loss.item()
            acc0 += calc_accuracy(y, labels0[:,0])
            acc1 += calc_accuracy(y, labels1[:,0])
            if train:
                if i != 0 and i % log_interval == 0:
                    print("Epoch {}: {} %,  loss: {}, acc0: {}, acc1: {}, ".format(
                        epoch,
                        round(100 * i / i_max),
                        loss_total / log_interval,
                        acc0 / log_interval,
                        acc1 / log_interval,
                        ))
                    loss_total, loss2_total, acc0, acc1, acc_g0, acc_g1 = [0.] * 6
        if not train:
            sum_batch = i_max
            eval_results = {
                    'loss': loss_total / sum_batch, 
                    'acc0': acc0 / sum_batch, 
                    'acc1': acc1 / sum_batch,
                    'loss2': loss2_total / sum_batch,
                    'acc_g0': acc_g0 / sum_batch,
                    'acc_g1': acc_g1 / sum_batch
                    }
            print("Test ", end='')
            print(*['{}: {}'.format(key, value) for key, value in eval_results.items()])
            loss_total, loss2_total, acc0, acc1, acc_g0, acc_g1 = [0.] * 6
            torch.save(model.state_dict(), SAVE_PATH/(model_type+'TF{}.pt'.format(epoch)))
            return eval_results

    with open(SAVE_PATH/(model_type+'TF.params'), 'w') as f:
        #json.dump({key: str(value) for key, value in params.items()}, f)
        json.dump(params, f, default=lambda x: str(x))
    
    eval_results_all = dict()
    for epoch in range(NUM_EPOCHS):
        model.train()
        run(train_loader, train=True)

        model.eval()
        with torch.no_grad():
            eval_results = run(test_loader, train=False)
        eval_results_all[epoch] = eval_results
        with open(SAVE_PATH/(model_type+'TFg0_result.json'.format(epoch)), 'w') as f:
            json.dump(eval_results_all, f)




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

def main():
    global SAVE_PATH, device
    parser = argparse.ArgumentParser()
    parser.add_argument('--human-mode', type=str, default='random')
    parser.add_argument('--model-type', type=str, default='Transformer_state_seq')
    #parser.add_argument('--memory', type=str, default='data/records/0211_record.pickle.gzip')
    parser.add_argument('--memory', type=str, default='data/records/0314_record.pickle.gzip')
    parser.add_argument('--remove-goal', action='store_true')
    parser.set_defaults(remove_goal=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()
    SEED = args.seed
    torch.manual_seed(SEED)
    SAVE_PATH /= str(SEED)
    SAVE_PATH.mkdir(exist_ok=True, parents=True)
    memory = utils.record.load(args.memory)
    device = torch.device('cuda:{}'.format(args.gpu))
    train_meta(memory, args.model_type, args.human_mode, args.remove_goal)


if __name__ == "__main__":
    main()
