import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import pandas as pd

import torch
from torch.utils.data import DataLoader

from train_utils.helper import construct_agent
from train_utils.dataset import UCI, AutoUCI

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

    # Parse configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    T = config['T']
    dim_context = config['dim_context']
    num_arm = config['num_arm']
    # ---------------- construct strategy -------------------------
    agent = construct_agent(config, device)

    # --------------- construct bandit ---------------------------
    # dataset = UCI(config['datapath'], dim_context, num_arm)
    dataset = AutoUCI(config['data_name'], dim_context, num_arm,
                      config['num_data'], config['version'])
    bandit = DataLoader(dataset, shuffle=True)
    # --------------------- training -----------------------------
    pbar = tqdm(range(T), dynamic_ncols=True, smoothing=0.1)
    loader = iter(bandit)
    reward_history = []
    accum_regret = 0

    for e in pbar:
        context, label = next(loader)
        context = context.squeeze(0).to(device)
        arm_to_pull = agent.choose_arm(context)
        # compute reward
        if label != arm_to_pull:
            reward = 0
        else:
            reward = 1
        # agent update
        agent.receive_reward(arm_to_pull, context[arm_to_pull], reward)
        agent.update_model(num_iter=min(e + 1, config['num_iter']))
        reward_history.append(reward)
        accum_regret += 1 - reward

        # save and upload results
        pbar.set_description(
            (
                f'Accumulated regret: {accum_regret}'
            )
        )
        if wandb and args.log:
            wandb.log(
                {
                    'Regret': accum_regret
                }
            )
    if wandb and args.log:
        run.finish()
    df = pd.DataFrame({'reward': reward_history})
    df.to_csv('log/uci-shuttle-history.csv')


if __name__ == '__main__':
    parser = ArgumentParser(description="basic paser for bandit problem")
    parser.add_argument('--config_path', type=str,
                        default='configs/uci/stat-shuttle-lmcts.yaml')
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()
    with open(args.config_path, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    for i in range(args.repeat):
        run(config, args)
    print('Done!')
