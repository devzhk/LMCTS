import torch


class Agent(object):
    def __init__(self, model,
                 optimizer,
                 criterion,
                 collector,
                 name='default'):
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
