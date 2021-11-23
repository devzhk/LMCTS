import torch
from .langevin import LangevinMC

from NeuralTS import transform


class LTS(object):
    def __init__(self, model, X,
                 criterion, num_iter=5,
                 eta=0.01, beta=0.1, sigma=1.0,
                 weight_decay=0.0,
                 bandit_generator=None,
                 unit_ball=False,
                 name='LMCTS'):
        '''
        Parameters: 
            - model: torch.nn.Module, the model to train
            - X: features Kxd tensor, 
            - eta: learning rate for the inner loop
            - beta: inverse temperature parameter
            - num_iter: number of iteration of inner loop
            - sigma: std for Langevine dynamic
            - weight_decay: weight of regularizer
            - bandit_generator: 
            - sigma: sigma for bandit generation
        '''
        self.model = model
        self.features = X
        self.device = X.device
        self.criterion = criterion
        self.num_iter = num_iter
        self.unit_ball = unit_ball
        self.optimizer = LangevinMC(
            self.model.parameters(), lr=eta, beta=beta,
            sigma=sigma, weight_decay=weight_decay)
        self.reg = weight_decay
        self.bandit_generator = bandit_generator
        self.itsname = name

    def clear(self):
        self.model.init_weights()
        self.Xtx = self.reg * torch.eye(self.features.shape[1])
        self.ChosenArms = []
        self.rewards = []
        self.cond = []

    def chooseArmToPlay(self):
        with torch.no_grad():
            pred = self.model(self.features)
        if self.unit_ball:
            arm_to_pull = self.model.fc.weight.data.clone().detach()
            self.ChosenArms.append(arm_to_pull)
        else:
            arm_to_pull = int(torch.argmax(pred))
            self.ChosenArms.append(self.features[arm_to_pull].tolist())
        return arm_to_pull

    def receiveReward(self, arm, reward):
        # accumulate reward
        self.rewards.append(reward)
        x = self.features[arm].view(1, -1)
        self.Xtx = self.Xtx + x.t() @ x
        cond_num = torch.linalg.cond(self.Xtx)
        self.cond.append(cond_num.item())

        data = torch.tensor(self.ChosenArms, device=self.device)
        true_reward = torch.tensor(self.rewards, device=self.device)

        self.model.train()
        for i in range(self.num_iter):
            pred = self.model(data)

            loss = self.criterion(pred.view(-1), true_reward)
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()

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
