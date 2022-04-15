import argparse
import logging
import os
import sys
import numpy as np
from pathlib import Path
import json

import torch
from torch import nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)
sys.path.append(str(ROOT_DIR))

import utils.record
import agents
from agents import train_params
from envs import goal_xs, map_goal_index
from utils.datasets import DoubleDataset2

logging.basicConfig(level=logging.INFO)

BATCH_SIZE = 128
NUM_EPOCHS = 20
NUM_WORKERS = 10

SAVE_PATH = Path('data/meta_models')

from train_meta import Human, calc_accuracy


def reward_based_loss(y, labels_all0, labels_all1, rewards, goals0, goals1, goals2=None, x=[]):
    # first utterance in 60 frames
    labels0 = labels_all0[:,0]
    labels1 = labels_all1[:,0]
    nll = -F.log_softmax(y, 1)

    scores_orig = [rewards[key].unsqueeze(1) for key in [2, 5, 8]]
    scores_orig = torch.hstack(scores_orig).to(device)
    # scores: (BATCH_SIZE, 3)

    logging.debug('label_mode: ' + label_mode)
    if label_mode == 'g0':
        # human's utterance is based on the agent's actual goal
        u_mask = nn.functional.one_hot(labels0, 3)
        g_prob = nn.functional.one_hot(goals0, 3)
    elif label_mode == 'g1':
        # agent knows the goal attributed by a human observer
        u_mask = nn.functional.one_hot(labels1, 3)
        g_prob = nn.functional.one_hot(goals1, 3).float()
    elif label_mode == 'g2':
        # based on agent's evel-2 inference of what goal a human attributes
        u_mask = nn.functional.one_hot(labels1, 3)
        g_prob = F.softmax(goals2.detach(), 1)
    elif label_mode == 'g10':
        # FOR ABLATION A
        # rocket and human share goal
        # reward not considered
        u_mask = nn.functional.one_hot(labels0, 3)
        loss = torch.mean(nll * u_mask)
        return loss
    elif label_mode == 'g100':
        # FOR ABLATION B
        # rocket and human DON'T share goal
        # agent assume it shares goal with human though goal_h != goal_a 
        u_mask = nn.functional.one_hot(labels1, 3)
        g_prob = nn.functional.one_hot(goals0, 3)
    elif label_mode == 'g101':
        # FOR ABLATION ?
        # rocket and human DON'T share goal
        # no rl
        u_mask = nn.functional.one_hot(labels1, 3)
        loss = torch.mean(nll * u_mask)
        return loss
    elif label_mode == 'g1000':
        # FOR ABLATION B
        # rocket and human DON'T share goal
        # supervised learning with false belief
        u_mask = nn.functional.one_hot(labels1, 3)
        loss = torch.mean(nll * u_mask)
        return loss
    else:
        raise AssertionError
    
    score_mode = 'softmax'
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

    scores *= g_prob
    scores = scores.sum(axis=1, keepdims=True)
    loss = torch.mean(nll * u_mask * scores)

    return loss

def goal_prediction_loss(y, labels_all0, labels_all1, rewards, goals0, goals1, goals2=None, x=[]):
    #g_prob = nn.functional.one_hot(goals0, 3)

    scores_orig = [rewards[key].unsqueeze(1) for key in [2, 5, 8]]
    scores_orig = torch.hstack(scores_orig).to(device)
    scores = F.softmax(scores_orig, 1) 
    #scores = F.softmax(scores_orig, 1) * g_prob
    #scores = scores.sum(axis=1, keepdims=True)

    labels1 = labels_all1[:,0]
    u_mask = nn.functional.one_hot(labels1, 3)
    match_rate = (F.softmax(y.detach(), 1) * u_mask).sum(axis=1, keepdims=True)
    nll2 = -F.log_softmax(goals2, 1)

    loss = torch.mean(nll2 * scores * match_rate)

    return loss

def save_gzip(data, path):
    import gzip
    import pickle
    with gzip.open(path, mode='wb', compresslevel=7) as f:
        f.write(pickle.dumps(data))

def load_gzip(path):
    import gzip
    import pickle
    with gzip.open(path, mode='rb') as f:
        data = pickle.load(f)
    return data


def train_meta(memory_path, model_type, human_mode, remove_goal, e_d_path, input_action=False):

    params = train_params[model_type]

    human = Human(human_mode)
    
    mem_name = Path(memory_path).stem
    e_d_name = Path(e_d_path).parent.stem
    a_sign = '-a' if input_action else ''
    cache_dir = Path('data/cache')
    cache_dir.mkdir(exist_ok=True, parents=True)
    train_dataset_path = cache_dir / 'train-{}-{}-{}{}.gzip'.format(mem_name, e_d_name, len_seq, a_sign)
    test_dataset_path = cache_dir / 'test-{}-{}-{}{}.gzip'.format(mem_name, e_d_name, len_seq, a_sign)
    dataset_ready = False
    try:
        train_dataset = load_gzip(str(train_dataset_path))
        test_dataset = load_gzip(str(test_dataset_path))
        dataset_ready = True
    except FileNotFoundError:
        pass
    if not dataset_ready:
        print('Cache not found')
        memory = utils.record.load(memory_path)
        bound = int(len(memory)*0.5)
        train_dataset = DoubleDataset2(memory[:bound], human, remove_goal=remove_goal,
                flatten=params['flatten'], info=True, device=device, 
                e_d_path=e_d_path, len_seq=len_seq, action=input_action)
        save_gzip(train_dataset, str(train_dataset_path))
        test_dataset = DoubleDataset2(memory[bound:], human, remove_goal=remove_goal,
                flatten=params['flatten'],  info=True, test=False, device=device, 
                e_d_path=e_d_path, len_seq=len_seq, action=input_action)
        save_gzip(test_dataset, str(test_dataset_path))
        dataset_ready = True
        print('Cached')

    train_dataset.test = True
    test_dataset.test = True

    print(len(train_dataset), len(test_dataset))
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=False)
    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=False)
    if input_action:
        print(train_dataset[0][0].shape)
        print(train_dataset[0][7].shape)
        sample_obs = F.one_hot(torch.tensor(train_dataset[0][7]), num_classes=4)
    else:
        sample_obs = train_dataset[0][0]

    # _goals2[i] ->  goals2[order[i]]
    # {0: [0, 0, 71], 1: [48, 16, 0], 2: [0, 70, 0]}
    order = TRANSLATER_ORDER #ground-truth: [2, 0, 1]
    translater = lambda x: x[:, order]

    # main model
    criterion = reward_based_loss
    n_in = params['calc_n_in'](sample_obs)
    model = params['model'](n_in=n_in, **params['kargs'])
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


    def run(data_loader, train=True):
        result_keys = ['loss', 'acc0', 'acc1', 'loss2', 'acc_g0', 'acc_g1']
        result = {key: 0. for key in result_keys}
        i_max = len(data_loader)
        log_interval = i_max // 4
        sum_batch = 0

        for i, data in enumerate(data_loader):
            num_batch = data[0].shape[0]
            sum_batch += num_batch
            inputs, labels0, labels1, rewards, goals0, goals1, _goals2 = data[:7]
            if input_action:
                inputs = data[7]
                inputs = F.one_hot(inputs, num_classes=4).float()
            inputs = params['pre_func'](inputs)
            inputs = inputs.to(device)
            labels0 = labels0.to(device)
            labels1 = labels1.to(device)
            goals0 = goals0.to(device)
            goals1 = goals1.to(device)
        
            if train:
                optimizer.zero_grad()

            y = model(inputs)

            goals2 = translater(_goals2).to(device)
            
            loss = criterion(y, labels0, labels1, rewards, goals0, goals1, goals2, inputs)
            loss2 = goal_prediction_loss(y, labels0, labels1, rewards, goals0, goals1, goals2, inputs)

            if train:
                loss.backward()
                optimizer.step()

            result['loss'] += loss.item()
            result['loss2'] += loss2.item()
            result['acc0'] += calc_accuracy(y, labels0[:,0]) * num_batch
            result['acc1'] += calc_accuracy(y, labels1[:,0]) * num_batch
            result['acc_g0'] += calc_accuracy(goals2, goals0) * num_batch
            result['acc_g1'] += calc_accuracy(goals2, goals1) * num_batch
            if train:
                if i != 0 and i % log_interval == 0:
                    for key  in result_keys:
                        result[key] /= sum_batch
                    print('Epoch: {} ({} %), order: {}'.format(epoch, round(100*i/i_max), order), end='')
                    print(result)
                    result = {key: 0. for key in result_keys}
                    sum_batch = 0
        if not train:
            for key  in result_keys:
                result[key] /= sum_batch
            print('Epoch: {} ({} %), order: {}'.format(epoch, 'test', order), end='')
            print(result)
            return result

    with open(SAVE_PATH/(model_type+'all.params'), 'w') as f:
        json.dump({key: str(value) for key, value in params.items()}, f)

    eval_results_all = dict()
    for epoch in range(NUM_EPOCHS):
        model.train()
        run(train_loader, train=True)

        model.eval()
        with torch.no_grad():
            eval_results = run(test_loader, train=False)

        eval_results_all[epoch] = eval_results
        if epoch == 19:
            torch.save(model.state_dict(), SAVE_PATH/(model_type+'last{}.pt'.format(epoch)))

        with open(SAVE_PATH/(model_type+'all_result.json'.format(epoch)), 'w') as f:
            json.dump(eval_results_all, f)



def main():
    global SAVE_PATH, TRANSLATER_ORDER, SEED, device, label_mode, len_seq
    parser = argparse.ArgumentParser()
    parser.add_argument('--human-mode', type=str, default='random')
    parser.add_argument('--model-type', type=str, default='Transformer_state_seq')
    #parser.add_argument('--memory', type=str, default='data/records/0211_record.pickle.gzip')
    parser.add_argument('--memory', type=str, default='data/records/0314_record.pickle.gzip')
    parser.add_argument('--remove-goal', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--translater', type=str)
    parser.add_argument('--e-d-path', type=str)
    parser.add_argument('--label-mode', type=str, default='g2')
    parser.add_argument('--len-seq', type=int, default=60)
    parser.add_argument('--input-action', action='store_true')
    parser.add_argument('--save-prefix', type=str, default='')

    parser.set_defaults(remove_goal=True)
    parser.set_defaults(input_action=False)
    args = parser.parse_args()

    label_mode = args.label_mode
    len_seq = args.len_seq
    SEED = args.seed
    device = torch.device('cuda:{}'.format(args.gpu))
    TRANSLATER_ORDER = [int(i) for i in args.translater]
    if args.save_prefix != '':
        SAVE_PATH /= args.save_prefix
    SAVE_PATH /= ''.join(map(str, TRANSLATER_ORDER))
    SAVE_PATH /= str(SEED)
    SAVE_PATH.mkdir(exist_ok=True, parents=True)

    train_meta(args.memory, args.model_type, args.human_mode, args.remove_goal, args.e_d_path,
            args.input_action)


if __name__ == "__main__":
    main()
