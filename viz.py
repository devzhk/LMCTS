# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
# plot regret
data = pd.read_csv('log/LinTS-regrets-5020.csv')
N = data['regret'].shape[0]
cummu_regret = np.zeros(N)

cummu_regret[0] = data['regret'][0]
for i in range(1, N):
    cummu_regret[i] = cummu_regret[i-1] + data['regret'][i]
line1 = plt.plot(data['Step'], cummu_regret)
plt.xlabel('Step')
plt.ylabel('Regret')
plt.title('Linear Thompson Sampling')
plt.savefig('figs/LinTS-sim-5020.png', bbox_inches='tight', dpi=400)
plt.show()

# %%
data = pd.read_csv('log/FTL-regrets-5020.csv')
N = data['regret'].shape[0]
cummu_regret = np.zeros(N)

cummu_regret[0] = data['regret'][0]
for i in range(1, N):
    cummu_regret[i] = cummu_regret[i-1] + data['regret'][i]
line1 = plt.plot(data['Step'], cummu_regret)
plt.show()

# %%
data = pd.read_csv('log/LMCTS-regrets-5020.csv')
N = data['regret'].shape[0]
cummu_regret = np.zeros(N)

cummu_regret[0] = data['regret'][0]
for i in range(1, N):
    cummu_regret[i] = cummu_regret[i-1] + data['regret'][i]
line1 = plt.plot(data['Step'], cummu_regret)
plt.xlabel('Step')
plt.ylabel('Regret')
plt.title('LMCTS (ours)')
plt.savefig('figs/LMCTS-5020.png', bbox_inches='tight', dpi=400)
plt.show()
plt.show()
# %%
