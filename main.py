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
from train_utils.dataset import UCI, Collector


def run(config, args):
    device = torch.device('cpu')
    # Load dataset
    dataset = UCI(config['data'])
    bandit = DataLoader(dataset, shuffle=True)
    # Define model
    model = LinearNet(config['data']['num_arm'],
                      config['data']['dim_context'])
    # create optimizer
    optimizer = LangevinMC(model.parameters(), lr=0.005, beta=0.5)
    # Define loss function
    criterion = torch.nn.MSELoss()
    collector = Collector()
    agent = Agent(model, optimizer, criterion, collector, name='LMCTS')
    pbar = range(config['train']['num_epochs'])
    loader = iter(bandit)
    reward_history = []
    for e in tqdm(pbar):
        context, label = next(loader)
        context = context.to(device)
        arm_to_pull = agent.choose_arm(context)
        if label != arm_to_pull:
            reward = 0
        else:
            reward = 1
        agent.receive_reward(arm_to_pull, context, reward)
        agent.update_model(num_iter=15)
        reward_history.append(reward)
        # pbar.set_description(
        #     (
        #         f'Epoch: {e}, accumulated reward: {sum(reward_history)}'
        #         f'Accumulated mean: {np.mean(reward_history)}'
        #     )
        # )
    df = pd.DataFrame({'reward': reward_history})
    df.to_csv('log/history.csv')


if __name__ == '__main__':
    parser = ArgumentParser(description="basic paser for bandit problem")
    parser.add_argument('--config_path', type=str,
                        default='configs/uci-stat-shuttle.yaml')
    parser.add_argument('--algo', type=str, default='LMC')
    args = parser.parse_args()
    with open(args.config_path, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    run(config, args)
    print('Done!')
