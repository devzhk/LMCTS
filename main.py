from train_utils.dataset import UCI, Collector
from algo.baselines import LinTS, FTL
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from models.classifier import LinearNet
from algo.langevin import LangevinMC
from algo.base import Agent

try:
    import wandb
except ImportError:
    wandb = None


def run(config, args):
    device = torch.device('cpu')
    T = config['bandit']['T']
    dim_context = config['bandit']['dim_context']
    num_arm = config['bandit']['num_arm']
    algo_name = config['train']['algo']

    # ---------------- construct strategy -------------------------
    if algo_name == 'LinTS':
        nu = 0.01 * np.sqrt(dim_context * np.log(T))
        agent = LinTS(num_arm, dim_context, nu, reg=1.0)
    elif algo_name == 'LMCTS':
        beta_inv = config['train']['beta_inv'] * dim_context * np.log(T)
        # Define model
        model = LinearNet(num_arm, dim_context, norm=True)
        # create optimizer
        optimizer = LangevinMC(model.parameters(), lr=config['train']['lr'],
                               beta_inv=beta_inv, weight_decay=2.0)
        # Define loss function
        criterion = torch.nn.MSELoss(reduction='sum')
        collector = Collector()
        agent = Agent(model, optimizer, criterion, collector, name='LMCTS')
    elif algo_name == 'FTL':
        agent = FTL(num_arm)
    # --------------- construct bandit ---------------------------
    dataset = UCI(config['datapath'], dim_context)
    bandit = DataLoader(dataset, shuffle=True)
    # --------------------- training -----------------------------
    pbar = range(T)
    loader = iter(bandit)
    reward_history = []
    accum_regret = 0
    if args.log and wandb:
        run = wandb.init(
            entity='hzzheng',
            project=config['log']['project'],
            group=config['log']['group'],
            config=config)
    for e in tqdm(pbar):
        context, label = next(loader)
        context = context.to(device)
        arm_to_pull = agent.choose_arm(context)
        # compute reward
        if label != arm_to_pull:
            reward = 0
        else:
            reward = 1
        # agent update
        agent.receive_reward(arm_to_pull, context, reward)
        agent.update_model(num_iter=min(e + 1, config['train']['num_iter']))
        reward_history.append(reward)
        accum_regret += 1 - reward
        # pbar.set_description(
        #     (
        #         f'Epoch: {e}, accumulated reward: {sum(reward_history)}'
        #         f'Accumulated mean: {np.mean(reward_history)}'
        #     )
        # )
        if wandb and args.log:
            wandb.log(
                {
                    'Regret': accum_regret
                }
            )
    if wandb and args.log:
        run.finish()
    df = pd.DataFrame({'reward': reward_history})
    df.to_csv('log/uci-history.csv')


if __name__ == '__main__':
    parser = ArgumentParser(description="basic paser for bandit problem")
    parser.add_argument('--config_path', type=str,
                        default='configs/uci-stat-shuttle.yaml')
    parser.add_argument('--log', action='store_true', default=True)
    args = parser.parse_args()
    with open(args.config_path, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    run(config, args)
    print('Done!')
