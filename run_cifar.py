import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import random

import torch
import multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from train_utils.losses import construct_loss
from train_utils.dataset import sample_data

from train_utils.helper import construct_agent_image


try:
    import wandb
except ImportError:
    wandb = None


def one_hot(img, num_arm):
    '''
    1x3x32x32 -> num_arm x 3 num_arm x 32 x 32
    '''
    cxt = torch.zeros((num_arm, 3 * num_arm, 32, 32), device=img.device)
    for i in range(num_arm):
        cxt[i, 3 * i: 3 * i + 3, :, :] = img[0]
    return cxt


def run(config, args):
    seed = random.randint(1, 10000)
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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    T = config['T']
    dim_context = config['dim_context']
    num_arm = config['num_arm']
    # ---------------- construct strategy -------------------------
    agent = construct_agent_image(config, device)

    # --------------- construct bandit ---------------------------
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    )

    dataset = CIFAR10('data', transform=transform, download=True)
    bandit = DataLoader(dataset, shuffle=True)
    # --------------------- training -----------------------------
    pbar = tqdm(range(T), dynamic_ncols=True, smoothing=0.1)
    loader = sample_data(bandit)
    reward_history = []
    accum_regret = 0

    for e in pbar:
        image, label = next(loader)
        context = one_hot(image, num_arm)
        context = context.to(device)
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
    print('Done!')


if __name__ == '__main__':
    parser = ArgumentParser(description="basic paser for bandit problem")
    parser.add_argument('--config_path', type=str,
                        default='configs/uci/shuttle-lmcts.yaml')
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

