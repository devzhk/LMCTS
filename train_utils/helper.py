import numpy as np


def randmax(arr):
    max_value = arr.max()
    idxs = [i for i, value in enumerate(arr) if value == max_value]
    return np.random.choice(idxs)


if __name__ == '__main__':
    import torch
    test_arr = torch.tensor([1.0, 4.0, 4.0, 2.0])
    print(randmax(test_arr))
