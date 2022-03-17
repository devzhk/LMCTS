import random
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import multiprocessing as mp

import torch
from torch.utils.data import DataLoader

from train_utils.helper import construct_agent_sim

from train_utils.dataset import SimData, sample_data
from train_utils.bandit import LinearBandit, QuadBandit, LogisticBandit


try:
    import wandb
except ImportError:
    wandb = None


def run(config, args):
    seed = random.randint(1, 10000)
    # seed = 2050
    print(f'Random seed: {seed}')
    torch.manual_seed(seed)
    if args.log and wandb:
        group = config['group'] if 'group' in config else None
        run = wandb.init(
            entity='hzzheng',
            project=config['project'],
            group=group,
            config=config)
        config = wandb.config

    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Parse argument
    data = torch.load(config['datapath'])
    theta = data['theta'].to(device)
    sigma = config['sigma']
    T = config['T']

    # Create bandit from dataset
    index = config['index'] if 'index' in config else 0
    num_data = config['num_data'] if 'num_data' in config else None
    dataset = SimData(config['datapath'], num_data=num_data, index=index)
    loader = DataLoader(dataset, shuffle=False)
    loader = sample_data(loader)
    if config['func'] == 'linear':
        bandit = LinearBandit(theta=theta, sigma=sigma)
    elif config['func'] == 'quad':
        bandit = QuadBandit(theta=theta, sigma=sigma)
    elif config['func'] == 'logistic':
        bandit = LogisticBandit(theta=theta, sigma=sigma)
    else:
        raise ValueError('Only linear or quadratic function')
    print(config)
    # ------------- construct strategy --------------------

    agent = construct_agent_sim(config, device)
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
    if wandb and args.log:
        run.finish()
    print('Done!')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser = ArgumentParser(description="basic paser for bandit problem")
    parser.add_argument('--config_path', type=str,
                        default='sweep/sweep-default.yaml')
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()
    with open(args.config_path, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    if args.repeat == 1:
        run(config, args)
    else:
        for i in range(args.repeat):
            p = mp.Process(target=run, args=(config, args))
            p.start()
