import math
import torch
from torch.optim import Optimizer


class LangevinMC(Optimizer):
    def __init__(self, 
                 params,        # parameters of the model  
                 lr=0.01,       # learning rate
                 beta=0.01,     # inverse temperature parameter
                 sigma=1.0,     # variance of the Gaussian noise
                 weight_decay=0.0):   # l2 penalty 
        if lr < 0:
            raise ValueError('lr must be positive')
        defaults = dict(lr=lr, beta=beta, sigma=sigma, weight_decay=weight_decay)
        super(LangevinMC, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            sigma = group['sigma']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is not None:
                    noise = math.sqrt(2 * lr / beta) * sigma * torch.randn(p.shape, device=p.device)
                    delta_p = - lr * p.grad + noise
                    if weight_decay != 0:
                        delta_p.add_(- p, alpha=weight_decay)
                    p.add_(delta_p)
        
