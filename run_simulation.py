import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from models.classifier import LinearNet
from algo.langevin import LangevinMC

from algo.base import SimLMCTS
from algo.baselines import LinTS, FTL

from train_utils.dataset import UCI, Collector, SimData
from train_utils.bandit import LinearBandit
try:
    import wandb
except ImportError:
    wandb = None


def run(config, args):
    if args.log and wandb:
        run = wandb.init(
            entity='hzzheng',
            project=config['log']['project'],
            group=config['log']['group'],
            config=config)

    device = torch.device('cpu')
    # Load dataset
    data = torch.load(config['datapath'])
    theta = data['theta']
    sigma = config['bandit']['sigma']
    T = config['bandit']['T']
    dim_context = config['bandit']['dim_context']
    num_arm = config['bandit']['num_arm']

    dataset = SimData(config['datapath'])
    loader = DataLoader(dataset, shuffle=False)
    loader = iter(loader)
    bandit = LinearBandit(theta=theta, sigma=sigma)
    print(config)
    # ------------- construct strategy --------------------
    algo_name = config['train']['algo']
    if algo_name == 'LinTS':
        nu = sigma * 0.01 * np.sqrt(dim_context * np.log(T))
        agent = LinTS(num_arm, dim_context, nu, reg=1.0)
    elif algo_name == 'LMCTS':
        beta_inv = config['train']['beta_inv'] * dim_context * np.log(T)

        print(f'Beta inverse: {beta_inv}')
        # Define model
        model = LinearNet(1, config['bandit']['dim_context'])
        # create optimizer
        optimizer = LangevinMC(model.parameters(), lr=config['train']['lr'],
                               beta_inv=beta_inv, weight_decay=2.0)

        # def lmc_func(x):
        #     if x > 100 and x % 300 == 0:
        #         return config['train']['lr'] / x
        #     else:
        #         return config['train']['lr']
        # scheduler = LambdaLR(optimizer, lr_lambda=lmc_func)
        # optimizer = SGD(model.parameters(),
        #                 lr=config['train']['lr'], weight_decay=1.0)
        # Define loss function
        criterion = torch.nn.MSELoss(reduction='sum')
        collector = Collector()
        agent = SimLMCTS(model, optimizer, criterion,
                         collector,
                         name='LMCTS')
    elif algo_name == 'FTL':
        agent = FTL(num_arm)
    # ---------------------------------------------------
    pbar = range(T)

    regret_history = []
    accum_regret = 0
    for e in tqdm(pbar):
        context = next(loader)
        context = context[0].to(device)
        arm_to_pull = agent.choose_arm(context)
        reward, regret = bandit.get_reward(context, arm_to_pull)
        agent.receive_reward(arm_to_pull, context[arm_to_pull], reward)
        agent.update_model(num_iter=min(e + 1, config['train']['num_iter']))
        regret_history.append(regret.item())
        accum_regret += regret.item()
        if wandb and args.log:
            wandb.log(
                {
                    'Regret': accum_regret
                }
            )
    df = pd.DataFrame({'regret': regret_history,
                       'Step': np.arange(config['bandit']['T'])}
                      )
    df.to_csv(f'log/{algo_name}-regrets-5020.csv')


if __name__ == '__main__':
    parser = ArgumentParser(description="basic paser for bandit problem")
    parser.add_argument('--config_path', type=str,
                        default='configs/gauss_bandit.yaml')
    parser.add_argument('--log', action='store_true', default=True)
    args = parser.parse_args()
    with open(args.config_path, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    run(config, args)
    print('Done!')
