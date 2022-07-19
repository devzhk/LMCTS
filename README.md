# Langevin Monte Carlo for Contextual Bandits (ICML2022)

This repository contains our pytorch implementation of Langevin Monte Carlo Thompson Sampling (LMC-TS), proposed in the paper [Langevin Monte Carlo for Contextual Bandits]()

_Abstract_:  Existing Thompson sampling-based algorithms need to construct a Laplace approximation of the posterior distribution, which is inefficient to sample in high dimensional applications for general covariance matrices. 
Moreover, the Gaussian approximation may not be a good surrogate for the posterior distribution for general reward generating functions. 
We propose an efficient posterior sampling algorithm, viz., Langevin Monte Carlo Thompson Sampling (LMC-TS), that uses Markov Chain Monte Carlo (MCMC) methods to directly sample from the posterior distribution in contextual bandits. 
Our method is computationally efficient since it only needs to perform noisy gradient descent updates without constructing the Laplace approximation of the posterior distribution. 

## Requirements
To install the necessary packages, run 
```bash
pip install -r requirements.txt
```

### Synthetic data
Run bandit algorithms on simulated bandit problems
```bash
python3 run_simulation.py --config_path configs/simulation/linear-LMCTS.yaml --repeat [number of experiments to repeat] 
```

You can add `--log` to turn on the wandb. 
Configuration file examples:
- Linear bandit: `configs/simulation/linear-LMCTS.yaml`
- Quadratic bandit: `configs/simulation/quad-LMCTS.yaml`
- Logistic bandit: `configs/simulation/logistic-LMCTS.yaml`

### UCI datasets
To run bandit algorithm on UCI datasets, use
```bash
python3 run_classifier.py --config_path configs/uci/shuttle-lmcts.yaml --repeat [number of experiments to repeat]
```
You can add `--log` to turn on the wandb. Configuration files are provided under folder `configs/uci`. 

### CIFAR10
To run bandit algorithm on CIFAR10 datasets, use
```bash
python3 run_cifar.py --config_path configs/image/cifar10-lmcts.yaml
```
Configuration files are provided under `configs/uci`.


## Code Structure
`run_cifar.py`, `run_classifier.py`, `run_simulation.py` are the main entries of the program, which contain the abstract code of bandit framework. 
These main entry files read configuration `.yaml` files from `configs` and parse them to run. 

`algo`,`train_utils`,`models` realize different modules in the framework.
More specifically, 
1. `algo` implements different online bandit algorithms. 
2. `train_utils` implements datasets, bandit instance, loss functions, and some helper functions.
3. `models` implements different model architectures. 

## Citation
```latex
@inproceedings{xu2022langevin,
  title={Langevin Monte Carlo for Contextual Bandits},
  author={Xu, Pan and Zheng, Hongkai and Mazumdar, Eric V and Azizzadenesheli, Kamyar and Anandkumar, Animashree},
  booktitle={International Conference on Machine Learning},
  pages={24830--24850},
  year={2022},
  organization={PMLR}
}
```