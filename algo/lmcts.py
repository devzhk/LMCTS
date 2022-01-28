import torch
from torch.utils.data import DataLoader
from .base import _agent
from train_utils.dataset import sample_data


class LMCTS(_agent):
    def __init__(self,
                 model,             # neural network model
                 optimizer,         # optimizer
                 criterion,         # loss function
                 collector,         # context and reward collector
                 batch_size=None,   # batchsize to update nn
                 decay_step=20,     # learning rate decay step
                 reduce=None,       # reduce update frequency
                 device='cpu',
                 name='default'):
        super(LMCTS, self).__init__(name)

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.collector = collector
        self.decay_step = decay_step
        if batch_size:
            self.loader = DataLoader(collector, batch_size=batch_size)
            self.batchsize = batch_size
        else:
            self.loader = None
            self.batchsize = None
        self.reduce = reduce
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
        if self.reduce:
            if self.step % self.reduce != 0:
                return
        self.model.train()
        # update using minibatch
        if self.batchsize and self.batchsize < self.step:
            if self.step % self.decay_step == 0:
                self.optimizer.lr = 10 * self.base_lr / self.step
            ploader = sample_data(self.loader)
            for i in range(num_iter):
                contexts, rewards = next(ploader)
                contexts = contexts.to(self.device)
                rewards = rewards.to(dtype=torch.float32, device=self.device)
                # rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                self.model.zero_grad()
                pred = self.model(contexts).squeeze(dim=1)
                loss = self.criterion(pred, rewards)
                loss.backward()
                self.optimizer.step()
            assert not torch.isnan(loss), "Loss is Nan!"
        else:
        # update using full batch
            if self.step % self.decay_step == 0:
                self.optimizer.lr = self.base_lr / self.step
            contexts, rewards = self.collector.fetch_batch()
            contexts = torch.stack(contexts, dim=0).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

            for i in range(num_iter):
                self.model.zero_grad()
                pred = self.model(contexts).squeeze(dim=1)
                loss = self.criterion(pred, rewards)
                loss.backward()
                self.optimizer.step()
            assert not torch.isnan(loss), "Loss is Nan!"

