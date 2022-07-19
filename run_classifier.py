import time
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import random

import torch
import multiprocessing as mp
from torch.utils.data import DataLoader

from torch.profiler import profile, record_function, ProfilerActivity

from train_utils.helper import construct_agent_cls
from train_utils.dataset import UCI, AutoUCI, sample_data

try:
    import wandb
except ImportError:
    wandb = None


def run(config, args):
    seed = random.randint(1, 10000)
    # seed = 2025
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
    print('Start running...')
    # Parse configuration
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    T = config['T']
    dim_context = config['dim_context']
    num_arm = config['num_arm']
    # ---------------- construct strategy -------------------------
    agent = construct_agent_cls(config, device)

    # --------------- construct bandit ---------------------------
    # dataset = UCI(config['datapath'], dim_context, num_arm)
    num_data = config['num_data'] if 'num_data' in config else None


    dataset = AutoUCI(config['data_name'], dim_context, num_arm,
                      num_data, config['version'])
    bandit = DataLoader(dataset, shuffle=True)
    # --------------------- training -----------------------------
    pbar = tqdm(range(T), dynamic_ncols=True, smoothing=0.1)
    loader = sample_data(bandit)
    reward_history = []
    accum_regret = 0

    # time
    choose_time = []
    update_time = []



    for e in pbar:
        context, label = next(loader)
        context = context.squeeze(0).to(device)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        # with torch.autograd.profiler.record_function('Arm selection'):
        # torch.cuda.synchronize(device)
        # start = time.time()
            with record_function("Arm selection"):
                arm_to_pull = agent.choose_arm(context)
        # torch.cuda.synchronize(device)
        # end = time.time()
        # select_cost = end - start
        # choose_time.append(select_cost)
        # compute reward
            if label != arm_to_pull:
                reward = 0
            else:
                reward = 1
            # agent update
            agent.receive_reward(arm_to_pull, context[arm_to_pull], reward)

        # torch.cuda.synchronize(device)
        # start = time.time()
            with record_function('Update model'):
                agent.update_model(num_iter=min(50, config['num_iter']))
        # torch.cuda.synchronize(device)
        # end = time.time()
        # update_cost = end - start
        # update_time.append(update_cost)

        reward_history.append(reward)
        accum_regret += 1 - reward
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        # save and upload results
        pbar.set_description(
            (
                f'Accumulated regret: {accum_regret}'
            )
        )
        if wandb and args.log:
            wandb.log(
                {
                    'Regret': accum_regret,
                    # 'Selecting time': select_cost,
                    # 'Update time': update_cost
                }
            )
    # print(f'averaged selecting cost: {sum(choose_time) / len(choose_time)}\n'
    #       f'averaged updating cost: {sum(update_time) / len(update_time)}\n')
    if wandb and args.log:
        run.finish()
    print('Done!')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser = ArgumentParser(description="basic paser for bandit problem")
    parser.add_argument('--config_path', type=str,
                        default='configs/uci/shuttle-lmcts.yaml')
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--log', action='store_true', default=False)
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

