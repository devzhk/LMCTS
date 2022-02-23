import math
import torch
from torch.optim import Optimizer

from torch import Tensor
from typing import List


def lmc(params: List[Tensor],
        d_p_list: List[Tensor],
        weight_decay: float,
        lr: float):
    r"""Functional API that performs Langevine MC algorithm computation.
    """

    for i, param in enumerate(params):
        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        param.add_(d_p, alpha=-lr)


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
        self.lr = lr
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
            lr = self.lr
            sigma = group['sigma']
            weight_decay = group['weight_decay']
            params_with_grad = []
            d_p_list = []
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    # noise = 0
                    noise = math.sqrt(2 * beta_inv / lr) * sigma * \
                        torch.randn(p.shape, device=p.device)
                    delta_p = p.grad
                    delta_p = delta_p.add(noise, alpha=-1.0)
                    d_p_list.append(delta_p)
                    # p.add_(delta_p)
            lmc(params_with_grad, d_p_list, weight_decay, lr)
