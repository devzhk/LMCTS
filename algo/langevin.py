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
            d_p = d_p.add_(param, alpha=weight_decay)

        param.add_(d_p, alpha=-lr)


class LangevinMC(Optimizer):
    def __init__(self,
                 params,              # parameters of the model
                 lr=0.01,             # learning rate
                 beta_inv=0.01,       # inverse temperature parameter
                 sigma=1.0,           # variance of the Gaussian noise
                 weight_decay=1.0,
                 device=None):   # l2 penalty
        if lr < 0:
            raise ValueError('lr must be positive')
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.beta_inv = beta_inv
        self.lr = lr
        self.sigma = sigma
        self.temp = - math.sqrt(2 * beta_inv / lr) * sigma
        self.curr_step = 0
        defaults = dict(weight_decay=weight_decay)
        super(LangevinMC, self).__init__(params, defaults)

    def init_map(self):
        self.mapping = dict()
        index = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    num_param = p.numel()
                    self.mapping[p] = [index, num_param]
                    index += num_param
        self.total_size = index

    @torch.no_grad()
    def step(self):
        self.curr_step += 1
        if self.curr_step == 1:
            self.init_map()

        lr = self.lr
        temp = self.temp
        noise = temp * torch.randn(self.total_size, device=self.device)

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            params_with_grad = []
            d_p_list = []
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)

                    start, length = self.mapping[p]
                    add_noise = noise[start: start + length].reshape(p.shape)
                    delta_p = p.grad
                    delta_p = delta_p.add_(add_noise)
                    d_p_list.append(delta_p)
                    # p.add_(delta_p)
            lmc(params_with_grad, d_p_list, weight_decay, lr)
