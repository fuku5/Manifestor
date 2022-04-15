import argparse
import time
import numpy as np
import sys
import torch
from torch import nn
import torch.nn.functional as F
import json
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)
sys.path.append(str(ROOT_DIR))

import agents
import utils.record
import utils.datasets
import train.train_meta
from agents import train_params

device = torch.device('cuda:0')


def calc_score(memory,  guesser_path, model_path, translate):
    
    bound = int(len(memory)*0.5)

    human = train.train_meta.Human('random')
    info = True
    params = agents.train_params['Transformer_state_seq']
    batch_size = 1
    dataset = utils.datasets.EpisodeWiseDataset(
            memory[:bound],
            human,
            flatten=params['flatten'],
            remove_goal=True,
            info=info,
            truncating=300)
    data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True)

    sample_obs = dataset[0][0]

    from evals.eval_meta import load_model
    n_in = 8
    model = load_model(params, model_path, n_in)
    model.eval()
    len_seq = 100
    encoder_param = agents.train_params['Guesser_Transformer']

    from train.train_e_d_guesser import build_model
    encoder, _, model_all = build_model(sample_obs)
    model_all.load_state_dict(torch.load(guesser_path, map_location=device))
    encoder = encoder.to(device)
    encoder.eval()

    to_list = lambda a: a.to('cpu').detach().tolist()
    results = list() 
    with torch.no_grad():
        for data in data_loader:
            x, u, r, g0, g1, masks = data[0], data[2], data[3], data[4], data[5], data[6]
            x_ = params['pre_func'](x)
            x_ = x_.to(device)
            masks = masks.to(device)
            u = u.to(device)

            g2 = encoder(x_, u, masks)
            g2 = translate(torch.softmax(g2, 1))
            
            x__ = torch.cat([x[:,i:i+len_seq] for i in range(x.shape[1] - len_seq + 1)])
            x__ = x__.transpose(1,0)
            x__ = x__.to(device)
            is_end = torch.where(masks.squeeze(0))[0]
            if len(is_end) != 0 and is_end[0] < x.shape[1] - len_seq + 1:
                end_idx = is_end[0]
            else:
                end_idx =  x.shape[1] - len_seq + 1
            y = model(x__[:, :end_idx])
            y = F.softmax(y, 1)
            u_ = F.one_hot(u.squeeze(0), 3)[:end_idx]
            #print(u, y.argmax(axis=1))
            masks = masks.squeeze(0)[:end_idx]
            match_rate = ((u_ * y).sum(axis=1) * (~masks))
            r_ = torch.vstack([torch.cat(r[key]) for key in [2,5,8]])
            #print(r_.sum(axis=1))
            #r_ = r_.sum(axis=1)
            #print(r_)
            attrs = dict()
            for key in ['g0','g1', 'match_rate', 'r_', 'g2']:
                attrs[key] = to_list(locals()[key])
            results.append(attrs)
            
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--guesser-path', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--len-seq', type=int, default=100)
    parser.add_argument('--translater', type=str)
    parser.add_argument('--memory', type=str, default='data/records/0314_record.pickle.gzip')
    parser.add_argument('--report-name', type=str)

    args = parser.parse_args()
    report_path_root = Path('data/meta_models/goal_loss')
    report_path_root.mkdir(exist_ok=True, parents=True)
    report_path = report_path_root / args.report_name
    
    model_path = args.model_path
    memory = utils.record.load(path=args.memory)
    
    translater_order = [int(i) for i in args.translater]
    translate = lambda x: x[:, translater_order]

    guesser_path = args.guesser_path
    device = torch.device('cuda:{}'.format(args.gpu))

    results = calc_score(memory, guesser_path, model_path, translate)
    with report_path.open('w') as f:
        json.dump(results, f)


