import torch

K = 20          # number of arms
d = 10          # dimensionality of context vectors
mu = 3.0        # mean of feature vectors

X = mu + torch.randn((K, d))
theta = torch.randn(d)

torch.save(
    {
        'X': X,
        'theta': theta
    },
    data_dir + 'linBdata-norm1.pt'
)
