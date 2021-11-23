import torch
from tqdm import tqdm

K = 50          # number of arms
d = 10          # dimensionality of context vectors
mu = 0.0        # mean of feature vectors
eps = 0.1      # biggest difference between best arm and worst arm
data_dir = 'data/'

X = torch.zeros((K, d))
theta = torch.rand(d)
X[0] = theta / torch.norm(theta)
for i in tqdm(range(K - 1)):
    diff = 1.0
    while diff > eps:
        xhat = torch.rand(d)
        xhat = xhat / torch.norm(xhat)
        diff = torch.abs(torch.dot(theta, theta) -
                         torch.dot(xhat, theta)) / torch.norm(theta)
    X[i+1, :] = xhat


torch.save(
    {
        'X': X,
        'theta': theta
    },
    data_dir + f'linBdata-{eps}.pt'
)


# TODO:
# 1. small difference
# 2. unit ball
# 3. orthonormal basis
# 4. update feature set
# 5. round robin warm up
