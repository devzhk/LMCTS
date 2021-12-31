import torch

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
                 scheduler=None,
                 name='default'):
        super(Agent, self).__init__(name)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.collector = collector
        self.base_lr = optimizer.lr
        self.step = 0

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
        # if self.step % 200 == 0:
        #     self.optimizer.lr = self.base_lr / np.sqrt(self.step)

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



        # if self.scheduler:
        #     self.scheduler.step()
