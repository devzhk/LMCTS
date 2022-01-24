import numpy as np
import torch
from torch.optim import SGD

from models.classifier import LinearNet, FCN
from models.conv import CNNModel
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
    batchsize = config['batchsize'] if 'batchsize' in config else None
    reduce = config['reduce'] if 'reduce' in config else None
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
                      collector,
                      name='LMCTS',
                      batch_size=batchsize,
                      reduce=reduce,
                      device=device)
    elif algo_name == 'EpsGreedy':
        agent = EpsGreedy(num_arm, config['eps'])
    elif algo_name == 'NeuralTS':
        model = FCN(1, dim_context * num_arm,
                    layers=config['layers'],
                    act=config['act'])
        model = model.to(device)
        optimizer = SGD(model.parameters(), lr=config['lr'], weight_decay=config['reg'])
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='mean')
        else:
            criterion = construct_loss(config['loss'], reduction='mean')
        collector = Collector()
        agent = NeuralTS(num_arm, dim_context,
                         model, optimizer,
                         criterion, collector,
                         config['nu'], reg=config['reg'],
                         batch_size=batchsize,
                         reduce=reduce,
                         device=device)
    elif algo_name == 'NeuralUCB':
        model = FCN(1, dim_context * num_arm,
                    layers=config['layers'],
                    act=config['act'])
        model = model.to(device)
        optimizer = SGD(model.parameters(), lr=config['lr'], weight_decay=config['reg'])
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='mean')
        else:
            criterion = construct_loss(config['loss'], reduction='mean')
        collector = Collector()
        agent = NeuralUCB(num_arm, dim_context,
                          model, optimizer,
                          criterion, collector,
                          config['nu'], reg=config['reg'],
                          batch_size=batchsize,
                          reduce=reduce,
                          device=device)
    elif algo_name == 'NeuralEpsGreedy':
        model = FCN(1, dim_context * num_arm,
                    layers=config['layers'],
                    act=config['act'])
        model = model.to(device)
        optimizer = SGD(model.parameters(), lr=config['lr'])
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='mean')
        else:
            criterion = construct_loss(config['loss'], reduction='mean')
        collector = Collector()
        agent = NeuralEpsGreedy(num_arm, dim_context,
                                model, optimizer,
                                criterion, collector,
                                config['eps'],
                                batch_size=batchsize,
                                reduce=reduce,
                                device=device)
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
    batchsize = config['batchsize'] if 'batchsize' in config else None
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
                      batch_size=batchsize,
                      device=device)
    elif algo_name == 'EpsGreedy':
        agent = EpsGreedy(num_arm, config['eps'])
    elif algo_name == 'NeuralTS':
        model = FCN(1, dim_context,
                    layers=config['layers'],
                    act=config['act'])
        model = model.to(device)
        optimizer = SGD(model.parameters(), lr=config['lr'], weight_decay=config['reg'])
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='mean')
        else:
            criterion = construct_loss(config['loss'], reduction='mean')
        collector = Collector()
        agent = NeuralTS(num_arm, dim_context,
                         model, optimizer,
                         criterion, collector,
                         config['nu'], reg=config['reg'],
                         device=device)
    elif algo_name == 'NeuralUCB':
        model = FCN(1, dim_context,
                    layers=config['layers'],
                    act=config['act'])
        model = model.to(device)
        optimizer = SGD(model.parameters(), lr=config['lr'], weight_decay=config['reg'])
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='mean')
        else:
            criterion = construct_loss(config['loss'], reduction='mean')
        collector = Collector()
        agent = NeuralUCB(num_arm, dim_context,
                          model, optimizer,
                          criterion, collector,
                          config['nu'], reg=config['reg'],
                          device=device)
    elif algo_name == 'NeuralEpsGreedy':
        model = FCN(1, dim_context,
                    layers=config['layers'],
                    act=config['act'])
        model = model.to(device)
        optimizer = SGD(model.parameters(), lr=config['lr'])
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='mean')
        else:
            criterion = construct_loss(config['loss'], reduction='mean')
        collector = Collector()
        agent = NeuralEpsGreedy(num_arm, dim_context,
                                model, optimizer,
                                criterion, collector,
                                config['eps'], device=device)
    else:
        raise ValueError(f'{algo_name} is not supported. Please choose from '
                         f'LinTS, LMCTS, NeuralTS, NeuralUCB, EpsGreedy')
    return agent


def construct_agent_image(config, device):
    dim_context = config['dim_context']
    num_arm = config['num_arm']
    algo_name = config['algo']
    T = config['T']
    batchsize = config['batchsize'] if 'batchsize' in config else None

    if algo_name == 'LMCTS':
        beta_inv = config['beta_inv'] * np.log(T)
        model = CNNModel(in_channel=3 * num_arm).to(device)
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
                      batch_size=batchsize,
                      reduce=4,
                      device=device)
    elif algo_name == 'NeuralTS':
        model = CNNModel(in_channel=3*num_arm).to(device)
        model = model.to(device)
        optimizer = SGD(model.parameters(), lr=config['lr'], weight_decay=config['reg'])
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='mean')
        else:
            criterion = construct_loss(config['loss'], reduction='mean')
        collector = Collector()
        agent = NeuralTS(num_arm, dim_context,
                         model, optimizer,
                         criterion, collector,
                         config['nu'],
                         batch_size=batchsize,
                         image=True,
                         reg=config['reg'],
                         reduce=10,
                         device=device)
    elif algo_name == 'NeuralUCB':
        model = CNNModel(in_channel=3 * num_arm).to(device)
        model = model.to(device)
        optimizer = SGD(model.parameters(), lr=config['lr'], weight_decay=config['reg'])
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='mean')
        else:
            criterion = construct_loss(config['loss'], reduction='mean')
        collector = Collector()
        agent = NeuralUCB(num_arm, dim_context,
                          model, optimizer,
                          criterion, collector,
                          config['nu'],
                          batch_size=batchsize,
                          image=True,
                          reduce=10,
                          reg=config['reg'],
                          device=device)
    elif algo_name == 'NeuralEpsGreedy':
        model = CNNModel(in_channel=3 * num_arm).to(device)
        model = model.to(device)
        optimizer = SGD(model.parameters(), lr=config['lr'])
        # Define loss function
        if 'loss' not in config:
            criterion = construct_loss('L2', reduction='mean')
        else:
            criterion = construct_loss(config['loss'], reduction='mean')
        collector = Collector()
        agent = NeuralEpsGreedy(num_arm, dim_context,
                                model, optimizer,
                                criterion, collector,
                                eps=config['eps'],
                                batch_size=batchsize,
                                reduce=5,
                                device=device)
    else:
        raise ValueError(f'Invalid algo name {algo_name}')
    return agent