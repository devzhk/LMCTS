# Langevin Monte Carlo Thompson Sampling


## Data description

### Synthetic data description

#### Linear bandit

$$\theta^\top x + \epsilon$$
We first generate $\theta^*\in\mathbb{R}^d$ with each coordinate randomly sampled from $N(0,1)$. Then we normalize $\theta^*$ such that $\|\theta^*\|_2=M$. For the feature vectors, we generate $X\in\mathbb{R}^{TK\times d}$ with each cooridnate randomly sampled from $N(0,1)$ and then normalize its rows that that $X[i,:]\in\mathbb{R}^d$ has norm $L$. 

During the training, at $t$-th iteration, action set is $X[(t-1)K+1,\ldots, tK; :]$, choose one action according the strategy.

Pull the chosen action, and received reward $r_t=\theta^{*\top}X[t_a;:]+\epsilon$, where the noise is sampled from $N(0,\sigma^2)$. 

**Hyperparameter search range**
- Linear Thompson Sampling: nu constant {1e-4, $1e-3$, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1}
- Lagevine Monte Carlo Thompson Sampling: lr: {0.1, **0.02**, 0.01}, beta: {0.5, 0.1, 0.05, **0.01**, 0.005, 0.001}, num_iter {1, 10, 15}.


#### Logistic Bandit

$$h(\theta^\top x) +\epsilon$$ where $h(v)=1/(1+\exp(v))$$

### UCI datasets
1. Statlog
2. CoverType
3. Magic
4. Mushroom

statlog: 
- LMCTS: lr {}
- LinTS: nu {0.1, 0.01, 0.02, 0.001}


### Yahoo R6A&B datasets

### Openbandit dataset

## Data generation and preprocessing 


## How to run bandit algorithm
### Synthetic data
To run bandit algorithm on synthetic data
```bash
python3 run_simulation.py --config_path configs/gaussian_bandit.yaml 
```

### UCI datasets


### TODO:
1. Squared reward (linear vs neural network)
2. Hyperparameter tuning for linear bandit
3. Repeated runs
4. Neural network for classfication (linear vs neural network)
   
Baseline: 
1. Neural Thompson Sampling
2. Neural UCB
3. LinTS