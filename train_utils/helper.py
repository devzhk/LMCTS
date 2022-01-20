import numpy as np
import torch
from torch.optim import SGD, Adam

from models.classifier import LinearNet, FCN
from algo.langevin import LangevinMC
from algo import LMCTS, LinTS, LinUCB, \
    EpsGreedy, NeuralTS, NeuralUCB, NeuralEpsGreedy

from train_utils.dataset import Collector

from .losses import construct_loss


def construct_agent_cls(config, device):
    '''
    Construct agent for classification task
    '''
    T = config['T']
    dim_context = config['dim_context']
    num_arm = config['num_arm']
    algo_name = config['algo']
    if algo_name == 'LinTS':
        nu = config['nu'] * np.sqrt(num_arm * dim_context * np.log(T))
        agent = LinTS(num_arm, num_arm * dim_context, nu, reg=1.0, device=device)
    elif algo_name == 'LinUCB':
        nu = lambda t: config['nu'] * np.sqrt(num_arm * dim_context * np.log(t))
        agent = LinUCB(num_arm, num_arm * dim_context, nu, reg=1.0, device=device)
    elif algo_name == 'LMCTS':
        beta_inv = config['beta_inv'] * dim_context * np.log(T)
        # Define model
        if config['model'] == 'linear':
            model = LinearNet(1, dim_context * num_arm)
        elif config['model'] == 'neural':
            model = FCN(1, dim_context * num_arm,
                        layers=config['layers'],
                        act=config['act'],
                        norm=True)
        model = model.to(device)
        # create Lagevine Monte Carol optimizer
        optimizer = LangevinMC(model.parameters(), lr=config['lr'],
                               beta_inv=beta_inv, weight_decay=2.0)
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='sum')
        else:
            criterion = construct_loss(config['loss'], reduction='sum')

        collector = Collector()
        agent = LMCTS(model, optimizer, criterion,
                      collector, name='LMCTS', device=device)
    elif algo_name == 'EpsGreedy':
        agent = EpsGreedy(num_arm, config['eps'])
    elif algo_name == 'NeuralTS':
        model = FCN(1, dim_context * num_arm,
                    layers=config['layers'],
                    act=config['act'])
        model = model.to(device)
        optimizer = SGD(model.parameters(), lr=config['lr'], weight_decay=config['reg'])
        criterion = torch.nn.MSELoss()
        collector = Collector()
        agent = NeuralTS(num_arm, dim_context,
                         model, optimizer,
                         criterion, collector,
                         config['nu'], m=config['layers'][0], reg=config['reg'],
                         device=device)
    elif algo_name == 'NeuralUCB':
        model = FCN(1, dim_context * num_arm,
                    layers=config['layers'],
                    act=config['act'])
        model = model.to(device)
        optimizer = SGD(model.parameters(), lr=config['lr'], weight_decay=config['reg'])
        criterion = torch.nn.MSELoss()
        collector = Collector()
        agent = NeuralUCB(num_arm, dim_context,
                          model, optimizer,
                          criterion, collector,
                          config['nu'], m=config['layers'][0], reg=config['reg'],
                          device=device)
    elif algo_name == 'NeuralEpsGreedy':
        model = FCN(1, dim_context * num_arm,
                    layers=config['layers'],
                    act=config['act'])
        model = model.to(device)
        optimizer = SGD(model.parameters(), lr=config['lr'])
        criterion = torch.nn.MSELoss()
        collector = Collector()
        agent = NeuralEpsGreedy(num_arm, dim_context,
                                model, optimizer,
                                criterion, collector,
                                config['eps'], device=device)
    else:
        raise ValueError(f'{algo_name} is not supported. Please choose from '
                         f'LinTS, LMCTS, NeuralTS, NeuralUCB, EpsGreedy')
    return agent


def construct_agent_sim(config, device):
    '''
    Construct agent for synthetic data (standard bandit setting)
    '''
    dim_context = config['dim_context']
    num_arm = config['num_arm']
    algo_name = config['algo']
    sigma = config['sigma']
    T = config['T']
    if algo_name == 'LinTS':
        nu = sigma * config['nu'] * np.sqrt(dim_context * np.log(T))
        agent = LinTS(num_arm, dim_context, nu, reg=1.0, device=device)
    elif algo_name == 'LinUCB':
        nu = lambda t: config['nu'] * np.sqrt(num_arm * dim_context * np.log(t))
        agent = LinUCB(num_arm, dim_context, nu, reg=1.0, device=device)
    elif algo_name == 'LMCTS':
        beta_inv = config['beta_inv'] * dim_context * np.log(T)

        print(f'Beta inverse: {beta_inv}')
        # Define model
        if config['model'] == 'linear':
            model = LinearNet(1, dim_context)
        elif config['model'] == 'neural':
            model = FCN(1, dim_context,
                        layers=config['layers'],
                        act=config['act'])
        model = model.to(device)
        # create optimizer
        optimizer = LangevinMC(model.parameters(), lr=config['lr'],
                               beta_inv=beta_inv, weight_decay=2.0)
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='sum')
        else:
            criterion = construct_loss(config['loss'], reduction='sum')
        collector = Collector()
        agent = LMCTS(model, optimizer, criterion,
                      collector,
                      name='LMCTS',
                      device=device)
    elif algo_name == 'EpsGreedy':
        agent = EpsGreedy(num_arm, config['eps'])
    elif algo_name == 'NeuralTS':
        model = FCN(1, dim_context,
                    layers=config['layers'],
                    act=config['act'])
        model = model.to(device)
        optimizer = SGD(model.parameters(), lr=config['lr'], weight_decay=config['reg'])
        criterion = torch.nn.MSELoss()
        collector = Collector()
        agent = NeuralTS(num_arm, dim_context,
                         model, optimizer,
                         criterion, collector,
                         config['nu'], m=config['layers'][0], reg=config['reg'],
                         device=device)
    elif algo_name == 'NeuralUCB':
        model = FCN(1, dim_context,
                    layers=config['layers'],
                    act=config['act'])
        model = model.to(device)
        optimizer = SGD(model.parameters(), lr=config['lr'], weight_decay=config['reg'])
        criterion = torch.nn.MSELoss()
        collector = Collector()
        agent = NeuralUCB(num_arm, dim_context,
                          model, optimizer,
                          criterion, collector,
                          config['nu'], m=config['layers'][0], reg=config['reg'],
                          device=device)
    elif algo_name == 'NeuralEpsGreedy':
        model = FCN(1, dim_context,
                    layers=config['layers'],
                    act=config['act'])
        model = model.to(device)
        optimizer = SGD(model.parameters(), lr=config['lr'])
        criterion = torch.nn.MSELoss()
        collector = Collector()
        agent = NeuralEpsGreedy(num_arm, dim_context,
                                model, optimizer,
                                criterion, collector,
                                config['eps'], device=device)
    else:
        raise ValueError(f'{algo_name} is not supported. Please choose from '
                         f'LinTS, LMCTS, NeuralTS, NeuralUCB, EpsGreedy')
    return agent

