# Langevin Monte Carlo Thompson Sampling


## Data description
Remark: for all experiments in this repo, bandit arm index always starts from 0. For example, K arms are indexed as 0, 1, ..., K-1.  
### Synthetic data description

#### Linear bandit

$$\theta^\top x + \epsilon$$
We first generate $\theta^*\in\mathbb{R}^d$ with each coordinate randomly sampled from $N(0,1)$. Then we normalize $\theta^*$ such that $\|\theta^*\|_2=M$. For the feature vectors, we generate $X\in\mathbb{R}^{TK\times d}$ with each cooridnate randomly sampled from $N(0,1)$ and then normalize its rows that that $X[i,:]\in\mathbb{R}^d$ has norm $L$. 

During the training, at $t$-th iteration, action set is $X[(t-1)K+1,\ldots, tK; :]$, choose one action according the strategy.

Pull the chosen action, and received reward $r_t=\theta^{*\top}X[t_a;:]+\epsilon$, where the noise is sampled from $N(0,\sigma^2)$.

#### Logistic Bandit

$$h(\theta^\top x) +\epsilon$$ where $h(v)=1/(1+\exp(v))$$

### UCI datasets
1. Statlog-shuttle
2. CoverType
3. Magic 
4. Adult 

## Data generation and preprocessing 


## How to run 
### Synthetic data
To run bandit algorithm on synthetic data, use `run_simulation.py`. 


```bash
python3 run_simulation.py --config_path configs/simulation/linear-LMCTS.yaml --repeat [number of experiments to repeat] --log 
```
Configuration file examples:
- Linear bandit: `configs/simulation/linear-LMCTS.yaml`
- Quadratic bandit: `configs/simulation/quad-LMCTS.yaml`
- Logistic bandit: `configs/simulation/logistic-LMCTS.yaml`


### UCI datasets
To run bandit algorithm on classification datasets, use
```bash
python3 run_classifier.py --config_path configs/uci/shuttle-lmcts.yaml --repeat [number of experiments to repeat] --log
```
### CIFAR10
```bash
python3 run_cifar.py --config_path configs/image/cifar10-lmcts.yaml
```


## Customize configuration file


## Hyperparameter Sweep 
We use wandb library to do grid search. `.yaml` files under sweep directory define the search space for each algorithm.  
Example: 
```bash
wandb sweep sweep/uci/shuttle-lmcts.yaml
wandb agent [agent id]
```

## Code Structure
`run_cifar.py`, `run_classifier.py`, `run_simulation.py` are the main entries of the program, which contain the abstract code of bandit framework. 
These main entry files read configuration `.yaml` files from `configs` and parse them to run. 

`algo`,`train_utils`,`models` realize different modules in the framework.
More specifically, 
1. `algo` implements different online bandit algorithms. 
2. `train_utils` implements datasets, bandit instance, loss functions, and some useful helper functions.
3. `models` implements different model architectures. 

