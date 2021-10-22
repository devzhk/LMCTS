import numpy as np
from scipy.special import comb


def entropy(p, n, k, z):
    sum_res = 0.0
    for i in range(k):
        sum_res += comb(n, i) * (p ** i) * (1-p) ** (n-i) * np.log2(z / (p** i * (1-p)** (n-i)))
    return sum_res / z


def compute(p, n, k):
    sum_res = 0
    for i in range(k):
        sum_res += comb(n, i) * (p ** i) * (1-p) ** (n-i)
    return sum_res


n = 100
k = 4

p = 0.005

z = compute(p, n, k)
# normalizing constant
print(z)

hx = entropy(p, n, k, z)
print(hx)