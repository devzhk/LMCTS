import math
import random
from threading import main_thread
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from bandits.BanditGenerator import SinBandit, ExpBandit, LinearBandit
from run_loop.Experiments import RunExpes
from algo.BanditBaselines import FTL, UCB, LinUCB, LinTS

from algo.LTS import LTS
from models import LinearModel

torch.manual_seed(2)
np.random.seed(2)

data_dir = 'data/'

mu = 3.0        # mean of feature vectors
sigma = 0.01     # std dev of noise

K = 20          # number of arms
d = 10          # dimensionality of context vectors


alpha = 2*(sigma**2)    # heuristics of UCB
Nexp = 1                # number of experiments to repeat
T = 5000                # number of time steps
# heuristic for Linear Thompson Sampling Strategy
nu = sigma * 1.0 * math.sqrt(d * np.log(T))
reg = 0.5               # regularization for Linear Thompson sampling


datafile = torch.load(data_dir + 'linBdata.pt')
X = datafile['X']
theta = datafile['theta']
# print(f'theta {theta}')

LinB = LinearBandit(X, theta=theta, sigma=sigma)
print(f'Linear Bandit means: \n {LinB.means}')


def beta_heuri(t):
    return sigma*np.sqrt(d * np.log(t))


strategy0 = FTL(K)
strategy1 = UCB(K, alpha)
strategy2 = LinUCB(X, beta_heuri)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = LinearModel(d, 1)
criterion = nn.MSELoss()

strategy3 = LinTS(X, nu, reg=reg)
strategy4 = LTS(model, torch.tensor(X).to(device), criterion,
                eta=0.01, num_iter=10, beta=0.2)

plt.figure(figsize=(10, 5))
RunExpes([strategy1, strategy3, strategy0],
         LinB, Nexp, T, 10, "off")
plt.title('Estimated mean regret through the time for the LinB problem')
plt.savefig('figs/test.png')


# if __name__ == '__main__':
#     parser = ArgumentParser(description='basic parser')
#     parser.add_argument('--data', type=str, default=None)
#     parser.add_argument('--K', type=int, default=20)
#     parser.add_argument('--d', type=int, default=10)
#     args = parser.parse_args()
