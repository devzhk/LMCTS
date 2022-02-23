from timeit import default_timer
import torch
import torch.nn as nn
from torch.optim import SGD

from torchvision.models import resnet18
from algo.langevin import LangevinMC


def random_sample_test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_round = 10000
    param_size = 5000

    # uniform distribution
    torch.cuda.synchronize()
    start = default_timer()

    for i in range(num_round):
        x = torch.rand(param_size, device=device)

    torch.cuda.synchronize()
    end = default_timer()
    print(f'Sample {num_round} rounds of {param_size}-dim sample from uniform dist: {end - start}')

    # Gaussian distribution
    torch.cuda.synchronize()
    start = default_timer()

    for i in range(num_round):
        x = torch.randn(param_size, device=device)

    torch.cuda.synchronize()
    end = default_timer()
    print(f'Sample {num_round} rounds of {param_size}-dim sample from Gaussian dist: {end - start}')


def update_test(optim, device):
    num_epoch = 1000

    model = resnet18().to(device)
    criterion = nn.CrossEntropyLoss()
    if optim == 'SGD':
        optimizer = SGD(model.parameters(), lr=0.001, weight_decay=0.01, momentum=0.9)
    elif optim == 'LMC':
        optimizer = LangevinMC(model.parameters(), lr=0.001, weight_decay=0.01)
    else:
        optimizer = None
    label = torch.randint(low=0, high=1000, size=(16,), device=device)
    image = torch.randn((16, 3, 224, 224), device=device)

    total = 0
    for i in range(num_epoch):
        model.zero_grad()
        pred = model(image)
        # _, preds = torch.max(pred, 1)

        loss = criterion(pred, label)
        loss.backward()
        torch.cuda.synchronize()
        start = default_timer()
        optimizer.step()
        torch.cuda.synchronize()
        end = default_timer()
        total += end - start

    print(f'Time cost: {total}')


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    update_test('SGD', device)
    update_test('LMC', device)
