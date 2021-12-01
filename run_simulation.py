import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from models.classifier import LinearNet
from algo.langevin import LangevinMC

from algo.base import SimLMCTS
from algo.baselines import LinTS, FTL

from train_utils.dataset import UCI, Collector, SimData
from train_utils.bandit import LinearBandit


def run(config, args):
    device = torch.device('cpu')
    # Load dataset
    data = torch.load(config['datapath'])
    theta = data['theta']
    sigma = config['bandit']['sigma']
    T = config['bandit']['T']
    dim_context = config['bandit']['dim_context']
    num_arm = config['bandit']['num_arm']

    dataset = SimData(config['datapath'])
    loader = DataLoader(dataset, shuffle=True)
    loader = iter(loader)
    bandit = LinearBandit(theta=theta, sigma=sigma)
    print(f'Running {args.algo}')
    # ------------- create strategy --------------------
    if args.algo == 'LinTS':
        nu = sigma * 0.001 * np.sqrt(dim_context * np.log(T))
        agent = LinTS(num_arm, dim_context, nu, reg=1.0)
    elif args.algo == 'LMCTS':
        # Define model
        model = LinearNet(1, config['bandit']['dim_context'])
        # create optimizer
        optimizer = LangevinMC(model.parameters(), lr=0.001,
                               beta=0.001, weight_decay=1.0)
        # Define loss function
        criterion = torch.nn.MSELoss()
        collector = Collector()
        agent = SimLMCTS(model, optimizer, criterion, collector, name='LMCTS')
    elif args.algo == 'FTL':
        agent = FTL(num_arm)
    # ---------------------------------------------------
    pbar = range(T)

    regret_history = []
    for e in tqdm(pbar):
        context = next(loader)
        context = context[0].to(device)
        arm_to_pull = agent.choose_arm(context)

        reward, regret = bandit.get_reward(context, arm_to_pull)
        agent.receive_reward(arm_to_pull, context[arm_to_pull], reward)
        agent.update_model(num_iter=config['train']['num_iter'])
        regret_history.append(regret.item())
        # pbar.set_description(
        #     (
        #         f'Epoch: {e}, accumulated reward: {sum(reward_history)}'
        #         f'Accumulated mean: {np.mean(reward_history)}'
        #     )
        # )
    df = pd.DataFrame({'regret': regret_history,
                       'Step': np.arange(config['bandit']['T'])}
                      )
    df.to_csv(f'log/{args.algo}-regrets.csv')


if __name__ == '__main__':
    parser = ArgumentParser(description="basic paser for bandit problem")
    parser.add_argument('--config_path', type=str,
                        default='configs/gauss_bandit.yaml')
    parser.add_argument('--algo', type=str, default='LMCTS')
    args = parser.parse_args()
    with open(args.config_path, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    run(config, args)
    print('Done!')
