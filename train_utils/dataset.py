
import numpy as np

import torch
from torch.utils.data import Dataset


class UCI(Dataset):
    def __init__(self, datacfg):
        super(UCI, self).__init__()
        self.loaddata(datacfg)

    def __getitem__(self, idx):
        return self.context[idx], self.label[idx]

    def __len__(self):
        return self.label.shape[0]

    def loaddata(self, cfg):
        data = np.loadtxt(cfg['datapath'])
        self.label = (data[:, -1] - 1).astype(int)
        # data preprocessing
        context = data[:, 0:cfg['dim_context']].astype(np.float32)
        # context = context - context.mean(axis=0, keepdims=True)
        self.context = context / np.linalg.norm(context, axis=1, keepdims=True)


class Collector(Dataset):
    '''
    Collect the context vectors that have appeared 
    '''

    def __init__(self):
        super(Collector, self).__init__()
        self.context = []
        self.rewards = []
        self.chosen_arms = []

    def __getitem__(self, key):
        return self.context[key], self.chosen_arms[key], self.rewards[key]

    def __len__(self):
        return len(self.rewards)

    def collect_data(self, context, arm, reward):
        self.context.append(context)
        self.chosen_arms.append(arm)
        self.rewards.append(reward)

    def fetch_batch(self, batch_size=None):
        if batch_size is None or batch_size > len(self.rewards):
            return self.context, self.chosen_arms, self.rewards
        else:
            offset = np.random.randint(0, len(self.rewards) - batch_size)
            return self.context[offset:offset + batch_size], self.rewards[offset: offset + batch_size]

    def clear(self):
        self.context = []
        self.rewards = []
        self.chosen_arms = []
