from sklearn.datasets import fetch_openml
import numpy as np

import torch
from torch.utils.data import Dataset


class SimData(Dataset):
    def __init__(self, datapath):
        data = torch.load(datapath)
        self.context = data['context']

    def __getitem__(self, idx):
        return self.context[idx]

    def __len__(self):
        return self.context.shape[0]


class UCI(Dataset):
    def __init__(self, datapath, dim_context, num_arms=2):
        super(UCI, self).__init__()
        self.dim_context = dim_context
        self.num_arms = num_arms
        self.loaddata(datapath, dim_context)

    def __getitem__(self, idx):
        x = self.context[idx]
        cxt = torch.zeros((self.num_arms, self.dim_context * self.num_arms))
        for i in range(self.num_arms):
            cxt[i, i * self.dim_context: (i + 1) * self.dim_context] = x
        return cxt, self.label[idx]

    def __len__(self):
        return self.label.shape[0]

    def loaddata(self, datapath, dim_context):
        data = np.loadtxt(datapath)
        self.label = (data[:, -1] - 1).astype(int)
        # data preprocessing
        context = data[:, 0:dim_context].astype(np.float32)
        # context = context - context.mean(axis=0, keepdims=True)
        self.context = context / np.linalg.norm(context, axis=1, keepdims=True)
        self.context = torch.tensor(self.context)


class AutoUCI(Dataset):
    def __init__(self, name, dim_context, num_arms, num_data=None, version='active'):
        super(AutoUCI, self).__init__()
        self.dim_context = dim_context
        self.num_arms = num_arms
        self.loaddata(name, version, num_data)

    def __getitem__(self, idx):
        x = self.context[idx]
        cxt = torch.zeros((self.num_arms, self.dim_context * self.num_arms))
        for i in range(self.num_arms):
            cxt[i, i * self.dim_context: (i + 1) * self.dim_context] = x
        return cxt, self.label[idx]

    def __len__(self):
        return self.label.shape[0]

    def loaddata(self, name, version, num_data):
        cxt, label = fetch_openml(name=name, version=version, data_home='data', return_X_y=True)
        self.label = np.array(label).astype(int) - 1
        context = np.array(cxt).astype(np.float32)
        if num_data:
            self.label = self.label[0:num_data]
            context = context[0:num_data, :]

        self.context = context / np.linalg.norm(context, axis=1, keepdims=True)
        self.context = torch.tensor(self.context)


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
