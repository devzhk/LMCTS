import torch
from .StochasticBandit import MAB
from .Arms import Gaussian, ZeroOne


'''
Library enabling to simulate contextual bandit problems + Eventually a classification problem
Code inspired by Emilie Kaufmann lab sessions : http://chercheurs.lille.inria.fr/ekaufman/SDM.html
'''


def LinearBandit(X, theta=None, sigma=0.001):
    '''Linear bandit model
    Args:
        X: matrix of features of dimension (K, d)
        sigma: stdev of Gaussian noise
    Return: 
        MAB instance
    '''
    Arms = []
    (K, d) = X.size()
    if theta is None:
        theta = torch.ones(d)
    mus = X @ theta
    for k in range(K):
        mu = mus[k]
        mu = mu.cpu().item()
        arm = Gaussian(mu, sigma**2)
        Arms.append(arm)

    return MAB(Arms)


def SinBandit(X, sigma=0.001):
    """X : matrix of features of dimension (K,d), 
    sigma : stdev of Gaussian noise"""
    Arms = []
    (K, d) = X.size()
    normalizing_factor = (1/d)

    for k in range(K):
        mu = normalizing_factor*torch.sum(torch.sin(X[k]))
        mu = mu.cpu().item()
        arm = Gaussian(mu, sigma**2)
        Arms.append(arm)

    return MAB(Arms)


def ExpBandit(X, sigma=0.5):
    """X : matrix of features of dimension (K,d), 
    sigma : stdev of Gaussian noise"""
    Arms = []
    (K, d) = X.size()
    normalizing_factor = (1/d)

    for k in range(K):
        mu = normalizing_factor*torch.sum(torch.exp(X[k])-torch.ones(d))
        mu = mu.cpu().item()
        arm = Gaussian(mu, sigma**2)
        Arms.append(arm)

    return MAB(Arms)


def TrickyBandit(X, sigma=0.5):
    """X : matrix of features of dimension (K,d), 
    sigma : stdev of Gaussian noise"""
    Arms = []
    (K, d) = X.size()
    normalizing_factor = (1/d)

    for k in range(K):
        mu = normalizing_factor*torch.sum(torch.sin(torch.pow(X[k], -1)))
        mu = mu.cpu().item()
        arm = Gaussian(mu, sigma**2)
        Arms.append(arm)

    return MAB(Arms)


def AbsBandit(X, sigma=0.5):
    """X : matrix of features of dimension (K,d), 
    sigma : stdev of Gaussian noise"""
    Arms = []
    (K, d) = X.size()

    for k in range(K):
        mu = torch.sum((torch.abs(X[k])))/d
        mu = mu.cpu().item()
        arm = Gaussian(mu, sigma**2)
        Arms.append(arm)

    return MAB(Arms)


def ClassBandit(y, K):
    """y : true_label 
       K : number of classes 
    """
    Arms = []
    for k in range(K):
        arm = ZeroOne(int((y == k)))
        Arms.append(arm)

    return MAB(Arms)
