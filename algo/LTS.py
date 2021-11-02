import torch
from .langevin import LangevinMC

from NeuralTS import transform


class LTS(object):
    def __init__(self, model, X,
                 criterion, num_iter=5,
                 eta=0.01, beta=0.1, sigma=1.0,
                 bandit_generator=None, name='LMCTS'):
        '''
        Parameters: 
            - model: torch.nn.Module, the model to train
            - X: features Kxd tensor, 
            - eta: learning rate for the inner loop
            - beta: inverse temperature parameter
            - num_iter: number of iteration of inner loop
            - bandit_generator: 
            - sigma: sigma for bandit generation
        '''
        self.model = model
        self.features = X
        self.device = X.device
        self.criterion = criterion
        self.num_iter = num_iter
        self.optimizer = LangevinMC(
            self.model.parameters(), lr=eta, beta=beta, sigma=sigma)

        self.bandit_generator = bandit_generator
        self.itsname = name

    def clear(self):
        self.model.init_weights()
        self.ChosenArms = []
        self.rewards = []

    def chooseArmToPlay(self):
        with torch.no_grad():
            pred = self.model(self.features)
        arm_to_pull = torch.argmax(pred)
        self.ChosenArms.append(self.features[arm_to_pull].tolist())
        return int(arm_to_pull)

    def receiveReward(self, arm, reward):
        # accumulate reward
        self.rewards.append(reward)

        data = torch.tensor(self.ChosenArms, device=self.device)
        true_reward = torch.tensor(self.rewards, device=self.device)

        self.model.train()
        for i in range(self.num_iter):
            pred = self.model(data)

            loss = self.criterion(pred.view(-1), true_reward)
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
        res = loss.item()

    def update_features(self):
        ''' 
        Update context features, 
        Required if update=True
        '''

        pass

    def new_MAB(self):
        self.update_features()
        bandit = self.bandit_generator(self.features, sigma=self.sigma)
        return bandit

    def name(self):
        return self.itsname
