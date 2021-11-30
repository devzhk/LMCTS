import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv('log/history.csv')
N = data['reward'].shape[0]
regret = np.zeros(N)

regret[0] = 1 - data['reward'][0]
for i in range(1, N):
    regret[i] = regret[i-1] + 1 - data['reward'][i]
line1 = plt.plot(data.index, regret)
plt.show()
