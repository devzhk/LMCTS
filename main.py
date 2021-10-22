import numpy as np
import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from bandits.BanditGenerator import SinBandit, ExpBandit
from run_loop.Experiments import RunExpes
from algo.BanditBaselines import FTL, UCB, LinUCB, LinTS

from algo.LTS import LTS
from models import LinearModel

K=20
d=10

#Generation of normalized features - ||x|| <= 1
X = torch.randn((K,d))
norms=torch.Tensor([torch.norm(x) for x in X])
X=X/torch.max(norms).item()

SinB=SinBandit(X)
ExpB=ExpBandit(X)

# print the means of the best two arms
print(np.sort(SinB.means)[-2:])
print(np.sort(ExpB.means)[-2:])

with open(f'X_{K}_{d}.pickle', 'wb') as setting:
    pickle.dump(X, setting)



# load the pickle file
K=[20]
d=[10]

for K,d in zip(K,d):
    with open(f'X_{K}_{d}.pickle', 'rb') as setting:
        X=pickle.load(setting)
        SinB=SinBandit(X)
        ExpB=ExpBandit(X)
        #We display the means of the arms by ascending order
        print(f'Contextual MAB K={K} d={d}')
        print('\n')
        print('SinB')
        print(np.sort(SinB.means))
        print('\n')
        print('ExpB')
        print(np.sort(ExpB.means))


sigma=0.5
nu=sigma
beta_heuri= lambda t:sigma*np.sqrt( d* np.log(t))
alpha=2*(sigma**2) 
Nexp=20
T=4000


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = LinearModel(10, 1)
criterion = nn.MSELoss(reduction='sum')


strategy0 = FTL(K)
strategy1 = UCB(K,alpha)
# strategy2 = LinUCB(X,beta_heuri)
# strategy3 = LinTS(X,nu)
strategy4 = LTS(model, torch.tensor(X).to(device), criterion)
plt.figure(figsize=(10,5))
RunExpes([strategy0, strategy1, strategy4],SinB,Nexp,T,10,"off")
plt.title('Estimated mean regret through the time for the SinB problem');
plt.savefig('test.png')