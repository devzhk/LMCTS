import torch

import yaml
from argparse import ArgumentParser


def generate(config):
    num_arm = config['num_arm']
    dim_context = config['dim_context']
    theta_norm = config['theta_norm']
    context_norm = config['context_norm']
    T = config['T']

    savepath = config['filename']
    # generate ground truth theta
    theta = torch.randn(dim_context)
    theta = theta / torch.norm(theta) * theta_norm
    # generate context data
    context = torch.randn((T, num_arm, dim_context))
    context = context / torch.norm(context, dim=2, keepdim=True) * context_norm

    torch.save(
        {
            'theta': theta,
            'context': context
        }, savepath
    )
    print('Data saved at {0}'.format(savepath))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='configs/data-linear.yaml')
    args = parser.parse_args()
    with open(args.config_path, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    generate(config)
