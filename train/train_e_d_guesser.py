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
import envs.wrappers
from envs import goal_xs, map_goal_index

logging.basicConfig(level=logging.INFO)

BATCH_SIZE = 32
NUM_EPOCHS = 20
NUM_WORKERS = 8

SAVE_PATH = Path('data/meta_models/')


from train.train_meta import Human, calc_accuracy
from utils.datasets import EpisodeWiseDataset



def build_model(sample_obs):
    encoder_param = train_params['Guesser_Transformer']
    encoder_n_in = encoder_param['calc_n_in'](sample_obs)
    encoder = encoder_param['model'](n_in=encoder_n_in, **(encoder_param['kargs']))
    
    decoder_param = train_params['MLP']
    decoder_n_in = encoder_param['calc_n_in'](sample_obs) + encoder_param['kargs']['n_out']
    decoder = decoder_param['model'](n_in=decoder_n_in, **(decoder_param['kargs']))

    model_all = nn.ModuleList([encoder, decoder])
    return encoder, decoder, model_all

def train_guesser(memory, model_type, human_mode, remove_goal):
    # train guesser that predicts g1 based on states and human utterances in a supervised way
    bound = int(len(memory)*0.5)

    params = train_params[model_type]

    human = Human(human_mode)
    # for eval how many episode is required

    train_dataset = EpisodeWiseDataset(memory[:bound], human, remove_goal=remove_goal,
            flatten=params['flatten'], info=True)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True)
    test_dataset = EpisodeWiseDataset(memory[bound:], human, remove_goal=remove_goal,
            flatten=params['flatten'],  info=True)
    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True)

    print(len(train_dataset), len(test_dataset))
    sample_obs = train_dataset[0][0]


    # model to guess P(g|s, u)
    criterion2 = nn.CrossEntropyLoss()

    encoder, decoder, model_all = build_model(sample_obs)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(model_all.parameters())

    def run(data_loader, train=True):
        result_keys = ['loss', 'acc0', 'acc1', 'loss2', 'acc_g0', 'acc_g1']
        result = {key: 0. for key in result_keys}
        #loss_total, loss2_total, acc0, acc1, acc_g0, acc_g1 = [0.] * 6
        i_max = len(data_loader)
        log_interval = i_max // 3
        sum_batch = 0

        for i, data in enumerate(data_loader):
            inputs, labels0, labels1, rewards, goals0, goals1, masks = data
            #print(goals1)
            num_batch = (~masks).sum().item()
            sum_batch += num_batch
            inputs = params['pre_func'](inputs)
            inputs = inputs.to(device)
            masks = masks.to(device)
            labels0 = labels0.to(device)
            labels1 = labels1.to(device)
            goals0 = goals0.to(device)
            goals1 = goals1.to(device)

            if train:
                optimizer.zero_grad()

            logging.debug('shape',(inputs.shape, masks.shape))
            goals2 = encoder(inputs, labels1, masks)
            goals2 = torch.softmax(goals2, 1)

            s, b, f = inputs.shape
            g2_ = goals2.unsqueeze(0).repeat(s, 1, 1)
            u1 = decoder(torch.cat([inputs, g2_], axis=2)).transpose(1, 0)
            masks_flatten = masks.reshape(-1)

            loss2 = criterion2(u1.reshape(-1,3)[~masks_flatten], labels1.reshape(-1)[~masks_flatten])

            if train:
                loss2.backward()
                optimizer.step()

            result['loss2'] += loss2.item()
            result['acc_g0'] += calc_accuracy(goals2, goals0) * num_batch
            result['acc_g1'] += calc_accuracy(u1.reshape(-1,3)[~masks_flatten], labels1.reshape(-1)[~masks_flatten]) * num_batch
            if train:
                if i != 0 and i % log_interval == 0:
                    for key  in result_keys:
                        result[key] /= sum_batch
                    print('Epoch: {} ({} %)'.format(epoch, round(100*i/i_max)), end='')
                    print(result)
                    result = {key: 0. for key in result_keys}
                    sum_batch = 0
        if not train:
            for key  in result_keys:
                result[key] /= sum_batch
            print('Epoch: {} ({} %)'.format(epoch, 'test'), end='')
            print(result)
            return result

    eval_results_all = dict()
    for epoch in range(NUM_EPOCHS):
        model_all.train()
        run(train_loader, train=True)

        model_all.eval()
        with torch.no_grad():
            eval_results = run(test_loader, train=False)

        eval_results_all[epoch] = eval_results
        torch.save(model_all.state_dict(), SAVE_PATH/('EncoderDecoder{}.pt'.format(epoch)))

        with open(SAVE_PATH/'EncoderDecoder_result.json'.format(epoch), 'w') as f:
            json.dump(eval_results_all, f)



def main():
    global SAVE_PATH, device
    parser = argparse.ArgumentParser()
    parser.add_argument('--human-mode', type=str, default='random')
    parser.add_argument('--model-type', type=str, default='Transformer_state_seq')
    parser.add_argument('--memory', type=str, default='data/records/0314_record.pickle.gzip')
    parser.add_argument('--remove-goal', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save-prefix', type=str, default='')
    parser.set_defaults(remove_goal=True)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.save_prefix != '':
        SAVE_PATH /= args.save_prefix
    SAVE_PATH /= str(args.seed)
    SAVE_PATH.mkdir(exist_ok=True, parents=True)
    device = torch.device('cuda:{}'.format(args.gpu))
    memory = utils.record.load(args.memory)
    train_guesser(memory, args.model_type, args.human_mode, args.remove_goal)


if __name__ == "__main__":
    main()
