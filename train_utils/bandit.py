import torch


class LinearBandit(object):
    def __init__(self, theta, sigma=1.0):
        self.theta = theta
        self.sigma = sigma

    def get_reward(self, X, arm):
        ''' 
        Args: 
            - X: shape of (num_arm, dim_context)
            - arm: int
        Return: 
            - reward: scalar

        '''
        prod = X @ self.theta
        regret = prod.max() - prod[arm]
        reward = prod[arm] + self.sigma * torch.randn(1, device=X.device)
        return reward, regret
