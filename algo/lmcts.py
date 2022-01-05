import torch
from .base import _agent


class LMCTS(_agent):
    def __init__(self,
                 model,             # neural network model
                 optimizer,         # optimizer
                 criterion,         # loss function
                 collector,         # context and reward collector
                 scheduler=None,    # learning rate scheduler
                 device='cpu',
                 name='default'):
        super(LMCTS, self).__init__(name)

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.collector = collector
        self.scheduler = scheduler
        self.step = 0
        self.base_lr = optimizer.lr
        self.device = device

    def clear(self):
        self.model.init_weights()
        self.collector.clear()
        self.step = 0

    def choose_arm(self, context):
        with torch.no_grad():
            pred = self.model(context)
            arm_to_pull = torch.argmax(pred)
        return int(arm_to_pull)

    def receive_reward(self, arm, context, reward):
        self.collector.collect_data(context, arm, reward)

    def update_model(self, num_iter=5):
        self.step += 1
        if self.step % 20 == 0:
            self.optimizer.lr = self.base_lr / self.step

        contexts, arms, rewards = self.collector.fetch_batch()
        contexts = torch.stack(contexts, dim=0)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        # TODO: adapt code for minibatch training
        self.model.train()
        for i in range(num_iter):
            self.model.zero_grad()
            pred = self.model(contexts).squeeze(dim=1)
            loss = self.criterion(pred, rewards)
            loss.backward()
            self.optimizer.step()
        assert not torch.isnan(loss), "loss is Nan"


# class LTS(object):
#     def __init__(self, model, X,
#                  criterion, num_iter=5,
#                  eta=0.01, beta=0.1, sigma=1.0,
#                  weight_decay=0.0,
#                  bandit_generator=None,
#                  unit_ball=False,
#                  name='LMCTS'):
#         '''
#         Parameters:
#             - model: torch.nn.Module, the model to train
#             - X: features Kxd tensor,
#             - eta: learning rate for the inner loop
#             - beta: inverse temperature parameter
#             - num_iter: number of iteration of inner loop
#             - sigma: std for Langevine dynamic
#             - weight_decay: weight of regularizer
#             - bandit_generator:
#             - sigma: sigma for bandit generation
#         '''
#         self.model = model
#         self.features = X
#         self.device = X.device
#         self.criterion = criterion
#         self.num_iter = num_iter
#         self.unit_ball = unit_ball
#         self.optimizer = LangevinMC(
#             self.model.parameters(), lr=eta, beta=beta,
#             sigma=sigma, weight_decay=weight_decay)
#         self.reg = weight_decay
#         self.bandit_generator = bandit_generator
#         self.itsname = name
#
#     def clear(self):
#         self.model.init_weights()
#         self.Xtx = self.reg * torch.eye(self.features.shape[1])
#         self.ChosenArms = []
#         self.rewards = []
#         self.cond = []
#
#     def chooseArmToPlay(self):
#         with torch.no_grad():
#             pred = self.model(self.features)
#         if self.unit_ball:
#             arm_to_pull = self.model.fc.weight.data.clone().detach()
#             self.ChosenArms.append(arm_to_pull)
#         else:
#             arm_to_pull = int(torch.argmax(pred))
#             self.ChosenArms.append(self.features[arm_to_pull].tolist())
#         return arm_to_pull
#
#     def receiveReward(self, arm, reward):
#         # accumulate reward
#         self.rewards.append(reward)
#         x = self.features[arm].view(1, -1)
#         self.Xtx = self.Xtx + x.t() @ x
#         cond_num = torch.linalg.cond(self.Xtx)
#         self.cond.append(cond_num.item())
#
#         data = torch.tensor(self.ChosenArms, device=self.device)
#         true_reward = torch.tensor(self.rewards, device=self.device)
#
#         self.model.train()
#         for i in range(self.num_iter):
#             pred = self.model(data)
#
#             loss = self.criterion(pred.view(-1), true_reward)
#             self.model.zero_grad()
#             loss.backward()
#             self.optimizer.step()
#
#     def update_features(self):
#         '''
#         Update context features,
#         Required if update=True
#         '''
#
#         pass
#
#     def new_MAB(self):
#         self.update_features()
#         bandit = self.bandit_generator(self.features, sigma=self.sigma)
#         return bandit
#
#     def name(self):
#         return self.itsname
