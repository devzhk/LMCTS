import math
import torch
from torch.optim import Optimizer


class LangevinMC(Optimizer):
    def __init__(self,
                 params,              # parameters of the model
                 lr=0.01,             # learning rate
                 beta_inv=0.01,       # inverse temperature parameter
                 sigma=1.0,           # variance of the Gaussian noise
                 weight_decay=1.0):   # l2 penalty
        if lr < 0:
            raise ValueError('lr must be positive')
        self.beta_inv = beta_inv
        self.curr_step = 0
        defaults = dict(lr=lr, sigma=sigma, weight_decay=weight_decay)
        super(LangevinMC, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        self.curr_step += 1
        # beta = \sqrt
        # beta = 1 / math.log(self.curr_step + 1) * self.beta
        beta_inv = self.beta_inv
        for group in self.param_groups:
            lr = group['lr']
            sigma = group['sigma']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is not None:
                    noise = math.sqrt(2 * lr * beta_inv) * sigma * \
                        torch.randn(p.shape, device=p.device)
                    delta_p = - lr * p.grad
                    if weight_decay != 0:
                        delta_p.add_(- p, alpha=2 * weight_decay)
                    p.add_(delta_p)
