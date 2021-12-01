# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
# plot regret
data = pd.read_csv('log/LinTS-regrets.csv')
N = data['regret'].shape[0]
cummu_regret = np.zeros(N)

cummu_regret[0] = data['regret'][0]
for i in range(1, N):
    cummu_regret[i] = cummu_regret[i-1] + data['regret'][i]
line1 = plt.plot(data['Step'], cummu_regret)
plt.show()

# %%
data = pd.read_csv('log/FTL-regrets.csv')
N = data['regret'].shape[0]
cummu_regret = np.zeros(N)

cummu_regret[0] = data['regret'][0]
for i in range(1, N):
    cummu_regret[i] = cummu_regret[i-1] + data['regret'][i]
line1 = plt.plot(data['Step'], cummu_regret)
plt.show()

# %%
data = pd.read_csv('log/LMCTS-regrets.csv')
N = data['regret'].shape[0]
cummu_regret = np.zeros(N)

cummu_regret[0] = data['regret'][0]
for i in range(1, N):
    cummu_regret[i] = cummu_regret[i-1] + data['regret'][i]
line1 = plt.plot(data['Step'], cummu_regret)
plt.show()
# %%
