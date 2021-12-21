import torch


class LinearBandit(object):
    def __init__(self, theta, sigma=1.0):
        self.theta = theta
        self.sigma = sigma

    def get_reward(self, X, arm):
        ''' 
        Args: 
            - X: contextual feature vector with shape (num_arm, dim_context)
            - arm: which arm to pull, int
        Return: 
            - reward: scalar
            - regret: expected regret

        '''
        prod = X @ self.theta
        regret = prod.max() - prod[arm]
        reward = prod[arm] + self.sigma * torch.randn(1, device=X.device)
        return reward, regret


class QuadBandit(object):
    def __init__(self, theta, sigma=1.0):
        self.theta = theta
        self.sigma = sigma

    def get_reward(self, X, arm):
        ''' 
        Args: 
            - X: contextual feature vector with shape (num_arm, dim_context)
            - arm: which arm to pull, int
        Return: 
            - reward: scalar
            - regret: expected regret

        '''
        prod = X @ self.theta
        h = 10 * prod ** 2
        reward = h[arm] + self.sigma * torch.randn(1, device=X.device)
        regret = h.max() - h[arm]
        return reward, regret
