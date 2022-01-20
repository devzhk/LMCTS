import torch
import torch.nn as nn


def NLL(pred, target):
    '''
    Negative log likelihood loss given x^\top theta

    Pred: x^\top \theta, (N, 1)
    '''
    loss = torch.sum(torch.log(1 + torch.exp(pred)) - target * pred)
    return loss


def construct_loss(name, reduction='mean'):
    if name == 'L1':
        func = nn.L1Loss(reduction=reduction)
    elif name == 'L2':
        func = nn.MSELoss(reduction=reduction)
    elif name == 'BCE':
        func = nn.BCEWithLogitsLoss(reduction=reduction)
    elif name == 'Logistic':
        func = NLL
    else:
        func = nn.MSELoss(reduction=reduction)
        print('Invalid loss function name. Setting loss function to MSELoss by default...')
    return func