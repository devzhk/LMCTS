import math
import yaml
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from bandits.BanditGenerator import SinBandit, ExpBandit, LinearBandit
from train_utils.Experiments import RunExpes
from algo.BanditBaselines import FTL, UCB, LinUCB, LinTS

from algo.lmcts import LTS
from models import LinearModel

torch.manual_seed(2)
np.random.seed(3)

data_dir = 'data/'


alpha = 0.1*(sigma**2)    # heuristics of UCB
Nexp = 20                # number of experiments to repeat
T = 5000                # number of time steps
# heuristic for Linear Thompson Sampling Strategy
nu = sigma * 0.1 * math.sqrt(d * np.log(T))
reg = 1.0               # regularization for Linear Thompson sampling


def beta_heuri(t):
    return 0.5 * sigma * np.sqrt(d * np.log(t))


def oneRound(X, theta, strategy, config):
    T = config['T']
    for i in tqdm(range(T)):
        bandit = LinearBandit(X[i], theta)
        


def run(config, args):
    num_arm = config['num_arm']
    sigma = config['sigma']
    theta_norm = config['theta_norm']
    context_norm = config['context_norm']
    datafile = torch.load(config['filename'])
    X = datafile['context']  # shape: T, num_arm, dim_context
    theta = datafile['theta']
    LinB = LinearBandit(X, theta, sigma=sigma)
    print(f'Linear Bandit means: \n {LinB.means}')
    T = config['T']

    strategy0 = FTL(num_arm)
    strategy1 = UCB(num_arm, alpha)
    strategy2 = LinUCB(X, beta_heuri, reg=reg)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = LinearModel(d, 1)
    criterion = nn.MSELoss()

    strategy3 = LinTS(Xhat, nu, reg=reg)
    strategy4 = LTS(model, Xhat.to(device), criterion,
                    eta=0.01, num_iter=10, beta=0.05, weight_decay=1.0)

    plt.figure(figsize=(10, 5))
    RunExpes([strategy1, strategy4, strategy2, strategy3, strategy0],
             LinB, Nexp, T, 10, "off")
    plt.title('Estimated mean regret through the time for the LinB problem')
    plt.savefig('figs/test.png')

    plt.figure().clear()
    t_arr = list(range(T))
    plt.plot(t_arr, strategy4.cond, label='cond of LMC-TS')
    plt.yscale('log')
    plt.savefig('figs/cond.png')


if __name__ == '__main__':
    parser = ArgumentParser(description='basic parser')
    parser.add_argument('--config_path', type=str,
                        default='configs/simulation.yaml')
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
