import time
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import random
from pathlib import Path

from obp.dataset import OpenBanditDataset



if __name__ == '__main__':
    parser = ArgumentParser(description="evaluate off-policy estimators.")
    parser.add_argument(
        "--behavior_policy",
        type=str,
        choices=["bts", "random"],
        required=True,
        help="behavior policy, bts or random.",
    )
    parser.add_argument(
        "--campaign",
        type=str,
        choices=["all", "men", "women"],
        required=True,
        help="campaign name, men, women, or all.",
    )
    args = parser.parse_args()
    dataset = OpenBanditDataset(#data_path=Path('data/openbandit'),
        behavior_policy=args.behavior_policy,
        campaign=args.campaign,
    )
    print(f'number of actions: {dataset.n_actions}\n '
          f'dimension of context: {dataset.dim_context}\n'
          f'Length of the recommendation list: {dataset.len_list}')
    bandit_feedback = dataset.obtain_batch_bandit_feedback()
    print(bandit_feedback['action_context'].shape)
    print(bandit_feedback['context'].shape)
    print(bandit_feedback['reward'].shape)


# 80 items, item 5
# 400000 user, context vector
# reward,