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
np.random.seed(3)

data_dir = 'data/'


sigma = 1.0     # std dev of noise

K = 50          # number of arms
d = 10          # dimensionality of context vectors


alpha = 0.1*(sigma**2)    # heuristics of UCB
Nexp = 20                # number of experiments to repeat
T = 5000                # number of time steps
# heuristic for Linear Thompson Sampling Strategy
nu = sigma * 0.1 * math.sqrt(d * np.log(T))
reg = 1.0               # regularization for Linear Thompson sampling


datafile = torch.load(data_dir + 'linBdata-0.02.pt')
X = datafile['X']
theta = datafile['theta']
# print(f'theta {torch.norm(theta)}')
norm_factor = torch.norm(X, dim=1, keepdim=True)
Xhat = X / norm_factor
LinB = LinearBandit(Xhat, theta=theta, sigma=sigma)
print(f'Linear Bandit means: \n {LinB.means}')


def beta_heuri(t):
    return 0.5 * sigma*np.sqrt(d * np.log(t))


strategy0 = FTL(K)
strategy1 = UCB(K, alpha)
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


# if __name__ == '__main__':
#     parser = ArgumentParser(description='basic parser')
#     parser.add_argument('--data', type=str, default=None)
#     parser.add_argument('--K', type=int, default=20)
#     parser.add_argument('--d', type=int, default=10)
#     args = parser.parse_args()
