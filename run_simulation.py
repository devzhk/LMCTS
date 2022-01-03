import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from models.classifier import LinearNet, FCN
from algo.langevin import LangevinMC

from algo import LMCTS
from algo.baselines import LinTS, FTL

from train_utils.dataset import Collector, SimData
from train_utils.bandit import LinearBandit, QuadBandit
try:
    import wandb
except ImportError:
    wandb = None


def run(config, args):
    if args.log and wandb:
        group = config['group'] if 'group' in config else None
        run = wandb.init(
            entity='hzzheng',
            project=config['project'],
            group=group,
            config=config)
        config = wandb.config

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Parse argument
    data = torch.load(config['datapath'])
    theta = data['theta'].to(device)
    sigma = config['sigma']
    T = config['T']
    dim_context = config['dim_context']
    num_arm = config['num_arm']

    # Create dataset
    dataset = SimData(config['datapath'])
    loader = DataLoader(dataset, shuffle=False)
    loader = iter(loader)
    if config['func'] == 'linear':
        bandit = LinearBandit(theta=theta, sigma=sigma)
    elif config['func'] == 'quad':
        bandit = QuadBandit(theta=theta, sigma=sigma)
    else:
        raise ValueError('Only linear or quadratic function')
    print(config)
    # ------------- construct strategy --------------------
    algo_name = config['algo']
    if algo_name == 'LinTS':
        nu = sigma * config['nu'] * np.sqrt(dim_context * np.log(T))
        agent = LinTS(num_arm, dim_context, nu, reg=1.0, device=device)
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
        criterion = torch.nn.MSELoss(reduction='sum')
        collector = Collector()
        agent = LMCTS(model, optimizer, criterion,
                         collector,
                         name='LMCTS',
                         device=device)
    elif algo_name == 'FTL':
        agent = FTL(num_arm)
    # ---------------------------------------------------
    pbar = tqdm(range(T), dynamic_ncols=True, smoothing=0.1)

    regret_history = []
    accum_regret = 0
    for e in pbar:
        context = next(loader)
        context = context[0].to(device)
        arm_to_pull = agent.choose_arm(context)
        reward, regret = bandit.get_reward(context, arm_to_pull)
        agent.receive_reward(arm_to_pull, context[arm_to_pull], reward)
        agent.update_model(num_iter=min(e + 1, config['num_iter']))
        regret_history.append(regret.item())
        accum_regret += regret.item()

        pbar.set_description(
            (
                f'Accumulative regret: {accum_regret}'
            )
        )
        if wandb and args.log:
            wandb.log(
                {
                    'Regret': accum_regret
                }
            )
    df = pd.DataFrame({'regret': regret_history,
                       'Step': np.arange(config['T'])}
                      )
    if wandb and args.log:
        run.finish()
    df.to_csv(f'log/{algo_name}-regrets-5020.csv')


if __name__ == '__main__':
    parser = ArgumentParser(description="basic paser for bandit problem")
    parser.add_argument('--config_path', type=str,
                        default='configs/sweep-default.yaml')
    parser.add_argument('--log', action='store_true', default=True)
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()
    with open(args.config_path, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    for i in range(args.repeat):
        run(config, args)
    print('Done!')
