import argparse
import time
import numpy as np
from pathlib import Path
import sys
import re
import torch
import torch.nn.functional as F
from torch import nn
import os
ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)
sys.path.append(str(ROOT_DIR))

import agents
import utils.record
from train.train_meta import Human 

len_seq = 100
fig = None
ax = None

def plot_state(state, alpha=1):
    ax[0].plot(state[0], state[1], 'o', color='cyan', alpha=alpha)
    return

def plot_bar(scores):
    ax[1].bar(range(3), scores[0])


def plot_states(states, y):
    ax[0].set_xlim(-1,1)
    ax[0].set_ylim(0,1)
    ax[1].set_ylim(0,1)
    
    alpha = 1
    discount_rate = 0.95
    alphas = discount_rate ** np.arange(len(states[0]))
    
    list(map(lambda x: plot_state(*x), zip(states[0, ::-1], alphas)))
    if False:
        for state in states[0, ::-1]:
            plot_state(state, alpha)
            alpha *= discount_rate
    plot_bar(y)
    return

def load_model(params, model_path, n_in):
    model = params['model'](n_in=n_in, **params['kargs'])
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device ))
    return model

def prepare_dataloader(memory, params, batch_size):
    bound = int(len(memory)*0.5)
    # use eval dataset

    human = Human('true_belief')
    info = False
    #dataset = utils.datasets.DatasetAction(
    dataset = utils.datasets.Dataset(
            memory[bound:],
            human,
            flatten=params['flatten'],
            remove_goal=True,
            info=info,
            len_seq=len_seq,
            truncating=200)
    data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True)
    return data_loader

def prepare(memory, mode, model_path, guesser_path, batch_size=1):
    params = agents.train_params[mode]
    data_loader = prepare_dataloader(memory, params, batch_size)

    # (60, 8)
    model = load_model(params, model_path, 8)
    
    if guesser_path is not None:
        n_in_guesser = 8 + 3
        guesser_param = agents.train_params['MLP']
        guesser = guesser_param['model'](n_in=n_in_guesser, **(guesser_param['kargs']))
        guesser = guesser.to(device)
    else:
        guesser = None
    return params, data_loader, model, guesser

def evaluate(memory, mode, model_path, guesser_path, visualize=False):
    from matplotlib import pyplot as plt
    from train.train_e_d_g_meta import goal_prediction_loss
    global fig, ax
    params, data_loader, model, guesser = prepare(memory, mode, model_path, guesser_path)

    if visualize:
        fig, ax = plt.subplots(ncols=2, figsize=(8,3))
        plt.ion()
        plt.show()

    model.eval()
    
    with torch.no_grad():
        for data in data_loader:
            ax[0].cla()
            ax[1].cla()
            if guesser is None:
                x = data[0]
                x_ = params['pre_func'](x)
                x_ = x_.to(device)
                y = model(x_)
                y = F.softmax(y)
            elif False:
                #x, u, g0, g1 = data[0], data[2], data[4], data[5]
                one_hot_label = nn.functional.one_hot(u, 3)
                one_hot_label = torch.cat([one_hot_label]*3, axis=2)
                x_ = torch.cat([x, one_hot_label], axis=2)
                x_ = x_.to(device)
                y = guesser(x_)
                y = F.softmax(y)
            else:
                inputs, _, labels1, rewards, goals0, goal1, _goals2 = data[:7]
                inputs = params['pre_func'](inputs)
                inputs = inputs.to(device)
                y = model(inputs)
                goals2 = translater(_goals2).to(device)
                loss2 = goal_prediction_loss(y, None, labels1, rewards, goals0, goals1, goals2, inputs)

            if visualize:
                plot_states(*map(tensor_to_numpy, (x, y)))
                fig.canvas.draw()

def tensor_to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def load_teachers(params, paths):
    import glob
    #paths = glob.glob('data/meta_models/teacher/0330_eval/*')
    #dirs = 'ed0/201/1000 ed1/201/1001 ed2/012/1002 ed3/120/1003 ed4/210/1004'.split()
    #dirs = ['data/meta_models/e_d_g_meta/ablation_v2_B_15000000/optimal/' + p for p in dirs]
    #paths = glob.glob('data/meta_models/e_d_g_meta/ablation_len_seq100/ablation_perfect/201/*')
    num_teachers = len(paths)
    teachers = [params['model'](n_in=8, **params['kargs']).to(device) for i in range(num_teachers)]
    for i, path in enumerate(paths):
        #path = Path(path) / 'Transformer_state_seqlast19.pt'
        teachers[i].load_state_dict(torch.load(path, map_location=device))
        teachers[i].eval()
    return teachers

def vote(x, teachers):
    ts = [tensor_to_numpy(F.softmax(teacher(x), 1)) for teacher in teachers]
    ts = [np.argmax(t, 1) for t in ts]
    ts = np.array(ts)
    #print(ts)
    NUM_ANSWER = 3
    votes = np.array([ts==i for i in range(NUM_ANSWER)])
    votes = votes.sum(axis=1)
    return votes.argmax(axis=0)

#def compare(memory, mode, model_path):
def compare(data_loader, model, teachers, params, out_dir=None, out_name=None):
    calc_t = True
    ys = list()
    ts = list()
    model.eval()
    with torch.no_grad():
        for x, label in data_loader:
            x = params['pre_func'](x)
            x = x.to(device)
            y = model(x)
            y = F.softmax(y, dim=1)
            y = tensor_to_numpy(y)
            ys.append(y)


            if calc_t:
                #t = teacher(x)
                #t = F.softmax(t, 1)
                #t = tensor_to_numpy(t)
                t = vote(x, teachers)
                t = np.array([tensor_to_numpy(F.softmax(teacher(x), 1)) for teacher in teachers]).transpose(1,0,2)
                ts.append(t)

    ys = np.concatenate(ys)
    if len(ts) != 0:
        ts = np.concatenate(ts)
    if out_dir is not None:
        import json
        #out_path = Path('data/meta_models/e_d_g_meta') / 'simulation100/comparison/' + out_name + '.json'
        #out_path = Path('data/meta_models/e_d_g_meta') / 'ablation_v2_B_15000000/comparison'
        out_dir.mkdir(exist_ok=True)
        if True:
            np.savez(str(out_dir / '{}.npz').format(out_name), ys=ys, ts=ts)
            return
        else:
            with open(out_dir / '{}.json'.format(out_name), 'w') as f:
                if calc_t:
                    json.dump(dict(ys=ys.tolist(), ts=ts.tolist()), f, indent=2)
                else:
                    json.dump(dict(ys=ys.tolist()), f, indent=2)
    ys_argmax = np.argmax(ys, axis=1)
    #ts_argmax = np.argmax(ts, axis=1)
    acc = (ys_argmax == ts).sum() / len(ys_argmax)
    print(acc)
    return acc


if __name__ == '__main__':
    global device
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--guesser-path', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--len-seq', type=int, default=100)
    parser.add_argument('--memory', type=str, default='data/records/0314_record.pickle.gzip')
    parser.add_argument('--out-dir', type=str, default='data/meta_models/e_d_g_meta/ablation_v2_B_15000000/comparison')
    parser.add_argument('--teacher-paths', type=str)
    args = parser.parse_args()
    len_seq = args.len_seq
    model_path = args.model_path
    device = torch.device('cuda:{}'.format(args.gpu))
    memory = utils.record.load(path=args.memory)

    params = agents.train_params['Transformer_state_seq']
    model = load_model(params, model_path, 8)
    data_loader = prepare_dataloader(memory, params, 128)
    teachers = load_teachers(params, args.teacher_paths.split(';'))
    out_dir = Path(args.out_dir)
    out_name = '_'.join(re.findall(r'([0-9]+)/([0-9]+)/[0-9]+', model_path)[0])
    compare(data_loader, model, teachers, params, out_dir, out_name)
