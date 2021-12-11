import torch
import numpy as np


class _agent(object):
    def __init__(self, name):
        self.name = name

    def clear(self):
        raise NotImplementedError

    def choose_arm(self, context):
        raise NotImplementedError

    def receive_reward(self, arm, context, reward):
        raise NotImplementedError

    def update_model(self, num_iter):
        raise NotImplementedError


class Agent(_agent):
    def __init__(self, model,
                 optimizer,
                 criterion,
                 collector,
                 name='default'):
        super(Agent, self).__init__(name)
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.collector = collector

    def clear(self):
        self.model.init_weights()
        self.collector.clear()

    def choose_arm(self, context):
        with torch.no_grad():
            pred = self.model(context)
            arm_to_pull = torch.argmax(pred)
        return int(arm_to_pull)

    def receive_reward(self, arm, context, reward):
        self.collector.collect_data(context, arm, reward)

    def update_model(self, num_iter=5):
        contexts, arms, rewards = self.collector.fetch_batch()
        contexts = torch.cat(contexts)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        idx = torch.tensor(arms).unsqueeze(1)
        # TODO: adapt code for minibatch training
        self.model.train()
        for i in range(num_iter):
            self.model.zero_grad()
            pred = self.model(contexts)
            pred_reward = torch.gather(pred, dim=1, index=idx).squeeze(-1)
            loss = self.criterion(pred_reward, rewards)
            loss.backward()
            self.optimizer.step()


class SimLMCTS(_agent):
    def __init__(self, model,
                 optimizer,
                 criterion,
                 collector,
                 name='default'):
        super(SimLMCTS, self).__init__(name)

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.collector = collector
        self.step = 0
        # self.beta_inv = optimizer.beta_inv

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
        self.optimizer.__setstate__({'lr': 0.04 / np.sqrt(self.step + 1)})
        # self.optimizer.beta_inv = np.log2(self.step + 2) * self.beta_inv

        contexts, arms, rewards = self.collector.fetch_batch()
        contexts = torch.stack(contexts, dim=0)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        # TODO: adapt code for minibatch training
        self.model.train()
        for i in range(num_iter):
            self.model.zero_grad()
            pred = self.model(contexts).squeeze(dim=1)
            loss = self.criterion(pred, rewards)
            loss.backward()
            self.optimizer.step()
        # noise =
        # self.model.fc.weight.data.add_()
