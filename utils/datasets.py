import torch
import numpy as np
import io
from PIL import Image, ImageQt

from envs import goal_xs

LEN_SEQ = 60

class Dataset(torch.utils.data.Dataset):
    #gamma = 0.995
    gamma = .995
    gammas = gamma ** np.arange(300).astype(np.float32)
    def __init__(self, memory, human, flatten=True, len_seq=LEN_SEQ,
            remove_goal=False, info=False, truncating=500, padding=False,
            with_image=False):
        assert (not with_image) or (with_image and ('img' in memory[0].keys()))
        super().__init__()
        self.flatten = flatten
        self.remove_goal = remove_goal
        self.info = info
        self.with_image = with_image 
        self.human = human
        self._init_lists()

        for episode in memory:
            human.update_belief()
            obs = np.array(episode['observation'], dtype=np.float32)
            labels0 = np.array(list([self.human.utter(o, 'true_belief') for o in obs]), dtype=np.long)
            labels1 = np.array(list([self.human.utter(o) for o in obs]), dtype=np.long)
            rwds = {key: [line['rewards'][key] for line in episode['info']] for key in goal_xs.keys()}
            if self.with_image:
                imgs = list(map(lambda png: np.asarray(Image.open(io.BytesIO(png))), 
                    episode['img']))
            n = (obs.shape[0] - 1) if padding else obs.shape[0] - len_seq
            for i in range(min(n, truncating)):
                if i <= obs.shape[0] - len_seq:
                    self.obss.append(obs[i:i+len_seq])
                    self.labels0.append(labels0[i:i+len_seq])
                    self.labels1.append(labels1[i:i+len_seq])
                else:
                    num_pad = i + len_seq - obs.shape[0]
                    self.obss.append(
                            np.pad(obs[i:i+len_seq], ((0, num_pad), (0, 0)), 'edge'))
                    num_pad = i + len_seq - labels0.shape[0]
                    self.labels0.append(
                            np.pad(labels0[i:i+len_seq], (0, num_pad), 'edge'))
                    self.labels1.append(
                            np.pad(labels1[i:i+len_seq], (0, num_pad), 'edge'))
                try:
                    assert self.obss[-1].shape[0] == len_seq
                    assert self.labels0[-1].shape[0] == len_seq
                    assert self.labels1[-1].shape[0] == len_seq
                except AssertionError as e:
                    print(i, len_seq, obs.shape, labels0.shape, labels1.shape, self.obss[-1].shape, self.labels1[-1].shape)
                    raise e
                if False:
                    rewards = dict()
                    for key in goal_xs.keys():
                        r = rwds[key][i:]
                        n = len(r)
                        rewards[key] = (r * self.gammas[:n]).sum()
                    self.rewards.append(rewards)
                else:
                    self.rewards.append({key: sum(rwds[key]) for key in goal_xs.keys()}) # kotti
                    #self.rewards.append({key: sum(rwds[key][i:i+len_seq]) for key in goal_xs.keys()})
                self.g0.append(self.human.believe(obs[i], 'true_belief'))
                self.g1.append(self.human.believe(obs[i]))
                if self.with_image and i < len(imgs):
                    self.imgs.append(imgs[i])
        self.g0 = np.array(self.g0)
        #print(0, [(self.g0==i).sum() for i in goal_xs.keys()])
        #print(0, [(self.g0==i).sum() for i in range(3)])
        self.labels0 = np.array(self.labels0)
        #print('l0', [(self.labels0==i).sum() for i in range(3)])
        self.labels1 = np.array(self.labels1)
        #print('l1', [(self.labels1==i).sum() for i in range(3)])

    def _init_lists(self):
        self.obss = list()
        # label based on agent's actual goal
        self.labels0 = list()
        # based on human's belief on agent's goal (level-1 inference)
        self.labels1 = list()
        self.rewards = list()
        # agent's actual goal
        self.g0 = list()
        # goal that human believes the agent has (level-1 inference)
        self.g1 = list()
        if self.with_image:
            self.imgs = list()


    def __getitem__(self, index):
        obs = self.obss[index]
        if self.remove_goal:
            obs = obs[:,:8]
        if self.flatten:
            obs = obs.flatten()

        rtn = [obs, self.labels0[index]]
        if self.info:
            rtn += [self.labels1[index], self.rewards[index], self.g0[index], self.g1[index]]
        if self.with_image:
            rtn += [self.imgs[index]]

        return rtn

    def __len__(self):
        return len(self.obss)


class EpisodeWiseDataset(torch.utils.data.Dataset):
    def __init__(self, memory, human, flatten=True, len_seq=LEN_SEQ,
            remove_goal=False, info=False, truncating=400):
        super().__init__()
        self.flatten = flatten
        self.remove_goal = remove_goal
        self.info = info
        self.human = human
        self.truncating = truncating
        self.len_seq = len_seq

        self.memory = [episode for episode in memory if len(episode['observation']) > 0]

    
    def __getitem__(self, index):
        self.human.update_belief()

        episode = self.memory[index]
        obs = np.array(episode['observation'][:self.truncating], dtype=np.float32)
        l = len(obs)
        obs = np.concatenate([obs, np.zeros((self.truncating-l, 9), dtype=np.float32)], axis=0)
        mask = np.zeros(self.truncating, dtype=np.bool)
        mask[l:] = True
        labels0 = np.array(list([self.human.utter(o, 'true_belief') for o in obs]), dtype=np.long)
        labels1 = np.array(list([self.human.utter(o) for o in obs]), dtype=np.long)
        rwds = {key: [line['rewards'][key] for line in episode['info']][:self.truncating]
                for key in goal_xs.keys()}
        for key in rwds.keys():
            if len(rwds[key]) < self.truncating:
                rwds[key] += [0.] * (self.truncating - len(rwds[key]))

        g0 = self.human.believe(obs[0], 'true_belief')
        g1 = self.human.believe(obs[0])

        if self.remove_goal:
            obs = obs[:,:8]
        if self.flatten:
            obs = obs.flatten()
        if self.info:
            #print(index, obs.shape, labels0.shape, labels1.shape, len(rwds[2]), g0, g1, mask.shape)
            if len(rwds[2]) != self.truncating:
                print(l, episode['info'])
                raise AssertionError(index)
            return obs, labels0, labels1, rwds, g0, g1, mask
        else:
            return obs, labels0

    def __len__(self):
        return len(self.memory)


class DoubleDataset(EpisodeWiseDataset):
    def __init__(self, memory, human, flatten=True, len_seq=LEN_SEQ,
            remove_goal=False, info=False, truncating=400, test=False, device=None):
        super().__init__(memory, human, flatten, len_seq, remove_goal,
                info, truncating)
        self.device = device
        self.indexes = np.array([0] + 
                [max(0, len(e['observation'][:truncating]) - self.len_seq) for e in self.memory])
        self.cum_indexes = np.cumsum(self.indexes)
        self.length = self.cum_indexes[-1]
        self.test = test
        #index = np.random.randint(0, min(l-self.len_seq, self.truncating))
        self._prepare_encoder_results()

    def _prepare_encoder_results(self):
        assert self.device is not None
        device = self.device
        self.goals2 = None
        from train_e_d_guesser import build_model
        sample_obs = self[0][0]
        encoder, _, encoder_decoder = build_model(sample_obs)
        encoder_decoder.load_state_dict(torch.load('data/meta_models/EncoderDecoder29.pt', map_location=device))
        encoder = encoder.to(device)
        encoder_decoder.eval()
        test = self.test
        self.test = False
        goals2 = list()
        from agents import train_params
        params = train_params['Transformer_state_seq']
        loader = torch.utils.data.DataLoader(
                self,
                batch_size=64,
                shuffle=False,
                num_workers=16,
                pin_memory=True)
        for data in loader:
            episode_wise = data[6:]
            inputs_episode, labels1_episode, masks = [episode_wise[i].to(device) for i in [0, 2, 6]]
            inputs_episode = params['pre_func'](inputs_episode)
            
            with torch.no_grad():
                _goals2 = encoder(inputs_episode, labels1_episode, masks)
            goals2.append(_goals2.detach())
        self.test = test
        self.goals2 = torch.cat(goals2).to('cpu')

    def __getitem__(self, index):
        #print(index, self.length)
        if self.test:
            episode_id = self.cum_indexes.searchsorted(index+1) - 1#np.argmax(index < self.cum_indexes, 0) - 1
            t_start = self.cum_indexes[episode_id]
            t = index - t_start
        else: 
            episode_id = index
            t = None 
        self.human.update_belief(episode_id % 3)
            
        
        episode = self.memory[episode_id]
        obs = np.array(episode['observation'][:self.truncating], dtype=np.float32)
        l = len(obs)
        obs = np.concatenate([obs, np.zeros((self.truncating-l, 9), dtype=np.float32)], axis=0)
        mask = np.zeros(self.truncating, dtype=np.bool)
        mask[l:] = True
        labels0 = np.array(list([self.human.utter(o, 'true_belief') for o in obs]), dtype=np.long)
        labels1 = np.array(list([self.human.utter(o) for o in obs]), dtype=np.long)
        rwds = {key: [line['rewards'][key] for line in episode['info']][:self.truncating]
                for key in goal_xs.keys()}
        for key in rwds.keys():
            if len(rwds[key]) < self.truncating:
                rwds[key] += [0.] * (self.truncating - len(rwds[key]))

        g0 = self.human.believe(obs[0], 'true_belief')
        g1 = self.human.believe(obs[0])

        if self.remove_goal:
            obs = obs[:,:8]
        if self.flatten:
            obs = obs.flatten()
        if self.info:
            episode_wise = [obs, labels0, labels1, rwds, g0, g1, mask]
        else:
            episode_wise = [obs, labels0]
        if self.goals2 is not None:
            episode_wise.append(self.goals2[episode_id])

        if t is None:
            t = np.random.randint(0, min(l-self.len_seq, self.truncating))
        partial = list(data[t:t+self.len_seq] for data in [obs, labels0, labels1])
        #print(index,episode_id, list([len(a) for a in partial]))
        if False:
            print(episode_id, t_start, t,
                    self.indexes[episode_id-2: episode_id+2],
                    self.cum_indexes[episode_id-2: episode_id+2])
        partial.append({key: sum(rwds[key]) for key in goal_xs.keys()})
        partial += [g0, g1]

        # partial(obs, labels0, labels1, rwds, g0, g1) + episode_wise(obs, labels0, labels1, rwds, g0, g1, mask)
        return partial + episode_wise

    def __len__(self):
        if self.test:
            return self.length
        else:
            return len(self.memory)

class DoubleDataset2(EpisodeWiseDataset):
    gammas = Dataset.gammas
    def __init__(self, memory, human, flatten=True, len_seq=LEN_SEQ,
            remove_goal=False, info=False, truncating=400, test=False, device=None, 
            e_d_path='A', action=False):
        super().__init__(memory, human, flatten, len_seq, remove_goal,
                info, truncating)
        self.device = device
        self.indexes = np.array([0] + 
                [max(0, len(e['observation'][:truncating]) - self.len_seq) for e in self.memory])
        self.cum_indexes = np.cumsum(self.indexes)
        self.length = self.cum_indexes[-1]
        self.test = test
        self.e_d_path = e_d_path
        self.action = action
        self.init()
        #index = np.random.randint(0, min(l-self.len_seq, self.truncating))

    def _load_encoder(self):
        import pathlib
        from train_e_d_guesser import build_model
        e_d_path = self.e_d_path
        device = self.device
        sample_obs = self[0][0]
        encoder, _, encoder_decoder = build_model(sample_obs)
        encoder_decoder.load_state_dict(torch.load(e_d_path, map_location=device))
        encoder = encoder.to(device)
        encoder_decoder.eval()
        from agents import train_params
        params = train_params['Transformer_state_seq']
        return encoder, params

    keys = ['obs', 'mask', 'labels0', 'labels1', 'rwds', 'g0', 'g1', 'g2', 'action'] 
    keys = [key + '_all' for key in keys]
    def init(self):
        self.init_done = False
        
        memory_processed = {key: list() for key in self.keys}

        for episode_id, episode in enumerate(self.memory):
            self.human.update_belief(episode_id % 3)
            obs = np.array(episode['observation'][:self.truncating], dtype=np.float32)
            l = len(obs)
            obs = np.concatenate([obs, np.zeros((self.truncating-l, 9), dtype=np.float32)], axis=0)
            mask = np.zeros(self.truncating, dtype=np.bool)
            mask[l:] = True
            labels0 = np.array(list([self.human.utter(o, 'true_belief') for o in obs]), dtype=np.long)
            labels1 = np.array(list([self.human.utter(o) for o in obs]), dtype=np.long)
            action = np.array(episode['action'][:self.truncating], dtype=np.long)
            l = len(action)
            action = np.concatenate([action, np.zeros(self.truncating-l, dtype=np.long)], axis=0)
            assert labels0.shape == action.shape
            #rwds = {key: [line['rewards'][key] for line in episode['info']][:self.truncating]
            rwds = {key: [line['rewards'][key] for line in episode['info']]
                    for key in goal_xs.keys()}
            for key in rwds.keys():
                if len(rwds[key]) < self.truncating:
                    rwds[key] += [0.] * (self.truncating - len(rwds[key]))

            g0 = self.human.believe(obs[0], 'true_belief')
            g1 = self.human.believe(obs[0])

            memory_processed['obs_all'].append(obs)
            memory_processed['mask_all'].append(mask)
            memory_processed['labels0_all'].append(labels0)
            memory_processed['labels1_all'].append(labels1)
            memory_processed['rwds_all'].append(rwds)
            memory_processed['g0_all'].append(g0)
            memory_processed['g1_all'].append(g1)
            memory_processed['g2_all'].append(None)
            memory_processed['action_all'].append(action)

        self.memory_processed = memory_processed
        encoder, encoder_params = self._load_encoder()
        test = self.test
        self.test = False
        device = self.device
        loader = torch.utils.data.DataLoader(
                self,
                batch_size=192,
                shuffle=False,
                num_workers=10,
                pin_memory=True)

        g2_all = list()
        for episode_wise in loader:
            inputs_episode, _, labels1_episode, _, _, _, masks_episode = episode_wise
            inputs_episode = inputs_episode.to(device)
            labels1_episode = labels1_episode.to(device)
            masks_episode = masks_episode.to(device)
            inputs_episode = encoder_params['pre_func'](inputs_episode)
            
            with torch.no_grad():
                _goals2 = encoder(inputs_episode, labels1_episode, masks_episode)
            g2_all.append(_goals2.detach())
        g2_all = torch.cat(g2_all).to('cpu')

        memory_processed['g2_all'] = g2_all
        self.test = test

        self.init_done = True

    def __getitem__(self, index):
        if self.test:
            episode_id = self.cum_indexes.searchsorted(index+1) - 1
            t_start = self.cum_indexes[episode_id]
            t = index - t_start
        else: 
            episode_id = index
            t = None 
        
        obs, mask, labels0, labels1, rwds, g0, g1, g2, action = \
                [self.memory_processed[key][episode_id] for key in self.keys]
        l = self.indexes[episode_id + 1]
        
        if self.remove_goal:
            obs = obs[:,:8]
        if self.flatten:
            obs = obs.flatten()
        if not self.init_done:
            rwds = {key: sum(rwds[key]) for key in goal_xs.keys()}
            if self.info:
                episode_wise = [obs, labels0, labels1, rwds, g0, g1, mask]
            else:
                episode_wise = [obs, labels0]
            return episode_wise

        if t is None:
            t = np.random.randint(0, l)
        #partial = list(data[t:t+self.len_seq] for data in [obs, labels0, labels1])
        #partial += [rwds, g0, g1, g2]
        partial = list(data[t:t+self.len_seq] for data in [obs, labels0, labels1])
        #print(rwds)
        #r_tmp = np.array([partial[-1][key] for key in rwds.keys()])
        #partial += [{key: sum(r[t:t+self.len_seq] ) for key, r in rwds.items()}]
        #print(partial[0][0][:2], partial[0][-1][:2], r_tmp, np.exp(r_tmp)/np.exp(r_tmp).sum())
        partial += [{key: (r[t:t+self.len_seq] * self.gammas[:self.len_seq]).sum() for key, r in rwds.items()}]
        partial += [g0, g1, g2]
        partial += [action[t:t+self.len_seq]]

        # partial(obs, labels0, labels1, rwds, g0, g1, g2) 
        # episode_wise(obs, labels0, labels1, rwds, g0, g1, mask)
        #print(index, episode_id, t, g2)
        return partial #+ episode_wise

    def __len__(self):
        if self.test:
            return self.length
        else:
            return len(self.memory)

class DatasetAction(Dataset):
    def __init__(self, memory, human, flatten=True, len_seq=LEN_SEQ,
            remove_goal=False, info=False, truncating=500, padding=False):
        super(Dataset, self).__init__()
        self.flatten = flatten
        self.remove_goal = remove_goal
        self.info = info
        self.with_image = 'img' in memory[0].keys()
        self.human = human
        self._init_lists()

        for episode in memory:
            human.update_belief()
            obs = np.array(episode['observation'], dtype=np.float32)
            action = torch.tensor(np.array(episode['action']))
            action = torch.nn.functional.one_hot(action, num_classes=4).float()
            labels0 = np.array(list([self.human.utter(o, 'true_belief') for o in obs]), dtype=np.long)
            labels1 = np.array(list([self.human.utter(o) for o in obs]), dtype=np.long)
            rwds = {key: [line['rewards'][key] for line in episode['info']] for key in goal_xs.keys()}
            if self.with_image:
                imgs = list(map(lambda png: np.asarray(Image.open(io.BytesIO(png))), 
                    episode['img']))
            n = (obs.shape[0] - 1) if padding else obs.shape[0] - len_seq
            for i in range(min(obs.shape[0] - len_seq - 1, truncating)):
                if i <= obs.shape[0] - len_seq:
                    self.obss.append(obs[i:i+len_seq])
                    self.labels0.append(labels0[i:i+len_seq])
                    self.labels1.append(labels1[i:i+len_seq])
                    self.actions.append(action[i:i+len_seq])
                else:
                    num_pad = i + len_seq - obs.shape[0]
                    self.obss.append(
                            np.pad(obs[i:i+len_seq], ((0, num_pad), (0, 0)), 'edge'))
                    self.labels0.append(
                            np.pad(labels0[i:i+len_seq], (0, num_pad), 'edge'))
                    self.labels1.append(
                            np.pad(labels1[i:i+len_seq], (0, num_pad), 'edge'))
                    self.actions.append(
                            np.pad(action[i:i+len_seq], (0, num_pad), 'edge'))
                if False:
                    rewards = dict()
                    for key in goal_xs.keys():
                        r = rwds[key][i:]
                        n = len(r)
                        rewards[key] = (r * self.gammas[:n]).suw()
                    self.rewards.append(rewards)
                self.rewards.append({key: sum(rwds[key]) for key in goal_xs.keys()}) # kotti
                #self.rewards.append({key: sum(rwds[key][i:i+len_seq]) for key in goal_xs.keys()})
                self.g0.append(self.human.believe(obs[i], 'true_belief'))
                self.g1.append(self.human.believe(obs[i]))
                if self.with_image:
                    self.imgs.append(imgs[i])
        self.g0 = np.array(self.g0)
        #print(0, [(self.g0==i).sum() for i in goal_xs.keys()])
        #print(0, [(self.g0==i).sum() for i in range(3)])
        self.labels0 = np.array(self.labels0)
        #print('l0', [(self.labels0==i).sum() for i in range(3)])
        self.labels1 = np.array(self.labels1)
        #print('l1', [(self.labels1==i).sum() for i in range(3)])

    def _init_lists(self):
        super()._init_lists()
        self.actions = list()

    def __getitem__(self, index):
        #obs = self.obss[index]
        #if self.remove_goal:
        #    obs = obs[:,:8]
        obs = self.actions[index]
        if self.flatten:
            obs = obs.flatten()

        rtn = [obs, self.labels0[index]]
        if self.info:
            rtn += [self.labels1[index], self.rewards[index], self.g0[index], self.g1[index]]
        if self.with_image:
            rtn += [self.imgs[index]]

        return rtn

    def __len__(self):
        return len(self.obss)
