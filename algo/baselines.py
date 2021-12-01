from re import L
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from .base import _agent
from train_utils.helper import randmax

'''
Linear Thompson Sampling method
'''


class LinTS(_agent):
    def __init__(self,
                 num_arm,       # number of arms
                 dim_context,   # dimension of context vector
                 nu,            #
                 reg=1.0,
                 name='Linear Thompson sampling'):
        super(LinTS, self).__init__(name)
        self.nu = nu
        self.reg = reg
        self.num_arm = num_arm
        self.dim_context = dim_context
        self.clear()

    def clear(self):
        self.t = 1
        # initialize the design matrix
        # self.Design = self.reg * torch.eye(self.dim_context)
        self.DesignInv = (1 / self.reg) * \
            torch.eye(self.dim_context)   # compute its inverse
        self.Vector = torch.zeros(self.dim_context)
        self.theta = torch.zeros(self.dim_context)
        self.last_cxt = 0
        self.last_reward = 0

    @torch.no_grad()
    def choose_arm(self, context):
        '''
        context: array of shape (num_arm, dim_context)
        '''
        tol = 1e-12
        if torch.linalg.det(self.DesignInv) < tol:
            cov = self.DesignInv + 0.001 * torch.eye(self.dim_context)
        else:
            cov = self.DesignInv
        dist = MultivariateNormal(self.theta.view(-1), self.nu ** 2 * cov)
        theta_tilda = dist.sample()
        arm_to_pull = torch.argmax(context @ theta_tilda).item()
        return arm_to_pull

    def receive_reward(self, arm, context, reward):
        self.last_cxt = context
        self.last_reward = reward

    def update_model(self, num_iter=None):
        self.Vector = self.Vector + self.last_reward * self.last_cxt
        omega = self.DesignInv @ self.last_cxt
        # update the inverse of the design matrix
        self.DesignInv = self.DesignInv - omega.view(-1, 1) @ omega.view(-1, 1).T / \
            (1 + torch.dot(omega, self.last_cxt))
        self.theta = self.DesignInv @ self.Vector
        self.t += 1


'''
Follow the leader strategy
'''


class FTL(_agent):
    def __init__(self, num_arm, name='FTL'):
        super(FTL, self).__init__(name)
        self.num_arm = num_arm
        self.clear()

    def clear(self):
        self.num_draw = torch.zeros(self.num_arm)
        self.rewards = torch.zeros(self.num_arm)

    def choose_arm(self, context):
        if self.num_draw.min() == 0:
            return randmax(- self.num_draw)
        else:
            return randmax(self.rewards)

    def receive_reward(self, arm, context, reward):
        num = self.num_draw[arm]
        self.rewards[arm] = (reward + num * self.rewards[arm]) / (num + 1)
        self.num_draw[arm] += 1

    def update_model(self, num_iter=None):
        pass


'''
Linear Upper Confidence Bound Bandit Algorithm
'''


class LinUCB(_agent):
    def __init__(self, beta, reg, name='LinUCB'):
        super(LinUCB, self).__init__(name)
        self.beta = beta
        self.reg = reg
