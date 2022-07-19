import PIL.GimpGradientFile
import torch
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from .base import _agent

from train_utils.dataset import sample_data

'''
Linear Thompson Sampling method
'''


class LinTS(_agent):
    def __init__(self,
                 num_arm,  # number of arms
                 dim_context,  # dimension of context vector
                 nu,  # temperature hyperparameter to control variance
                 reg=1.0,  # regularization weight
                 device='cpu',  # device
                 name='Linear Thompson sampling'):
        super(LinTS, self).__init__(name)
        self.nu = nu
        self.reg = reg
        self.num_arm = num_arm
        self.dim_context = dim_context
        self.device = device
        self.clear()

    def clear(self):
        self.t = 1
        # initialize the design matrix
        # self.Design = self.reg * torch.eye(self.dim_context)
        self.DesignInv = (1 / self.reg) * \
                         torch.eye(self.dim_context,
                                   device=self.device)  # compute its inverse
        self.Vector = torch.zeros(self.dim_context, device=self.device)
        self.theta = torch.zeros(self.dim_context, device=self.device)
        self.last_cxt = 0
        self.last_reward = 0

    @torch.no_grad()
    def choose_arm(self, context):
        '''
        context: array of shape (num_arm, dim_context)
        '''
        tol = 1e-12
        if torch.linalg.det(self.DesignInv) < tol:
            cov = self.DesignInv + 0.00001 * \
                  torch.eye(self.dim_context, device=self.device)
        else:
            cov = self.DesignInv
        dist = MultivariateNormal(self.theta.view(-1), self.nu ** 2 * cov)
        theta_tilda = dist.sample()
        arm_to_pull = torch.argmax(context @ theta_tilda).item()
        return arm_to_pull

    def receive_reward(self, arm, context, reward):
        self.last_cxt = context
        self.last_reward = reward

    def update_model(self, num_iter=None):
        self.Vector = self.Vector + self.last_reward * self.last_cxt
        omega = self.DesignInv @ self.last_cxt
        # update the inverse of the design matrix
        self.DesignInv = self.DesignInv - omega.view(-1, 1) @ omega.view(-1, 1).T / \
                         (1 + torch.dot(omega, self.last_cxt))
        self.theta = self.DesignInv @ self.Vector
        self.t += 1


'''
Linear UCB algorithm 

'''


class LinUCB(_agent):
    def __init__(self,
                 num_arm,  # number of arms
                 dim_context,  # dimension of context vector
                 nu,  # temperature hyperparameter to control variance
                 reg=1.0,  # regularization weight
                 device='cpu',  # device
                 name='Linear UCB'):
        super(LinUCB, self).__init__(name)
        self.nu = nu
        self.reg = reg
        self.num_arm = num_arm
        self.dim_context = dim_context
        self.device = device
        self.clear()

    def clear(self):
        self.t = 1
        # initialize the design matrix
        # self.Design = self.reg * torch.eye(self.dim_context)
        self.DesignInv = (1 / self.reg) * \
                         torch.eye(self.dim_context,
                                   device=self.device)  # compute its inverse
        self.Vector = torch.zeros(self.dim_context, device=self.device)
        self.theta = torch.zeros(self.dim_context, device=self.device)
        self.last_cxt = 0
        self.last_reward = 0

    @torch.no_grad()
    def choose_arm(self, context):
        '''
        context: array of shape (num_arm, dim_context)
        '''
        tol = 1e-12
        if torch.linalg.det(self.DesignInv) < tol:
            cov = self.DesignInv + 0.00001 * \
                  torch.eye(self.dim_context, device=self.device)
        else:
            cov = self.DesignInv

        norms = [context[i].T @ cov @ context[i] for i in range(self.num_arm)]

        pred = context @ self.theta + self.nu * \
               torch.tensor(norms, device=self.device)
        arm_to_pull = torch.argmax(pred).item()
        return arm_to_pull

    def receive_reward(self, arm, context, reward):
        self.last_cxt = context
        self.last_reward = reward

    def update_model(self, num_iter=None):
        self.Vector = self.Vector + self.last_reward * self.last_cxt
        omega = self.DesignInv @ self.last_cxt
        # update the inverse of the design matrix
        self.DesignInv = self.DesignInv - omega.view(-1, 1) @ omega.view(-1, 1).T / \
                         (1 + torch.dot(omega, self.last_cxt))
        self.theta = self.DesignInv @ self.Vector
        self.t += 1


'''
Epsilon greedy algorithm
'''


def randmax(arr):
    '''
    Randomly pick an element among maximal elements.
    '''
    max_value = arr.max()
    idxs = [i for i, value in enumerate(arr) if value == max_value]
    return np.random.choice(idxs)


class EpsGreedy(_agent):
    '''
    epsilon-greedy
    '''

    def __init__(self, num_arm, eps=0.0, name='eps-greedy'):
        super(EpsGreedy, self).__init__(name)
        self.num_arm = num_arm
        self.eps = eps
        self.clear()

    def clear(self):
        self.num_draw = torch.zeros(self.num_arm)
        self.rewards = torch.zeros(self.num_arm)
        self.step = 0

    def choose_arm(self, context):
        if self.num_draw.min() == 0:
            return randmax(- self.num_draw)
        else:
            prob = self.eps / np.sqrt(self.step)
            if np.random.uniform() < prob:
                return np.random.randint(self.num_arm)
            else:
                return randmax(self.rewards)

    def receive_reward(self, arm, context, reward):
        num = self.num_draw[arm]
        self.rewards[arm] = (reward + num * self.rewards[arm]) / (num + 1)
        self.num_draw[arm] += 1

    def update_model(self, num_iter=None):
        self.step += 1


'''
Neural Thompson Sampling

Adapted from the official implementation at https://github.com/ZeroWeight/NeuralTS/blob/master/learner_neural.py
1. the authors use the inverse of the diagonal elements of U to approximate the design matrix inverse U^{-1}
2. We divide the  according to paper's algorithm
'''


def get_param_size(model):
    '''
    Args:
        model: nn.Module
    Return:
        the number of learnable parameters in the model
    '''
    num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return num


class NeuralTS(_agent):
    def __init__(self,
                 num_arm,  # number of arms
                 dim_context,  # dimension of context feature
                 model,  # Neural network model
                 optimizer,  # optimizer
                 criterion,  # loss function
                 collector,  # context and reward collector
                 nu,  # exploration variance
                 batch_size=None,  # batchsize to update nn
                 image=False,  # image dataset?
                 reduce=None,  # reduce update frequency
                 reg=1.0,  # regularization weight, lambda in original paper
                 device='cpu',
                 name='NeuralTS'):
        super(NeuralTS, self).__init__(name)
        self.image = image
        self.reduce = reduce
        self.num_arm = num_arm
        self.dim_context = dim_context

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        if batch_size:
            self.loader = DataLoader(collector, batch_size=batch_size)
            self.batchsize = batch_size
        else:
            self.loader = None
            self.batchsize = None
        self.collector = collector
        self.nu = nu
        self.reg = reg
        self.device = device
        self.step = 0

        self.num_params = get_param_size(model)
        self.clear()

    def clear(self):
        # self.model.init_weights()
        self.collector.clear()
        self.Design = self.reg * \
                      torch.ones(self.num_params, device=self.device)
        self.last_cxt = 0
        self.step = 0

    def choose_arm(self, context):
        rewards = []
        for i in range(self.num_arm):
            self.model.zero_grad()
            if self.image:
                ri = self.model(context[i:i + 1, :, :, :])
            else:
                ri = self.model(context[i])
            ri.backward()

            grad = torch.cat([p.grad.contiguous().view(-1).detach()
                              for p in self.model.parameters()])

            squared_sigma = self.reg * self.nu * grad * grad / self.Design
            sigma = torch.sqrt(torch.sum(squared_sigma))

            sample_r = ri + torch.randn(1, device=self.device) * sigma
            rewards.append(sample_r.item())
        arm_to_pull = np.argmax(rewards)
        return arm_to_pull

    def receive_reward(self, arm, context, reward):
        self.collector.collect_data(context, arm, reward)
        self.last_cxt = context

    def update_model(self, num_iter):
        self.step += 1
        if self.reduce:
            if self.step % self.reduce != 0:
                return

        for p in self.optimizer.param_groups:
            p['weight_decay'] = self.reg / self.step

        # update using minibatch
        if self.batchsize and self.batchsize < self.step:
            ploader = sample_data(self.loader)
            for i in range(num_iter):
                contexts, rewards = next(ploader)
                contexts = contexts.to(self.device)
                rewards = rewards.to(dtype=torch.float32, device=self.device)
                self.model.zero_grad()
                pred = self.model(contexts).squeeze(dim=1)
                loss = self.criterion(pred, rewards)
                loss.backward()
                self.optimizer.step()
            assert not torch.isnan(loss), 'Loss is Nan!'
        else:
            # update using full batch
            contexts, rewards = self.collector.fetch_batch()
            contexts = torch.stack(contexts, dim=0).to(self.device)
            rewards = torch.tensor(
                rewards, dtype=torch.float32, device=self.device)
            self.model.train()
            for i in range(num_iter):
                self.model.zero_grad()
                pred = self.model(contexts).squeeze(dim=1)
                loss = self.criterion(pred, rewards)
                loss.backward()
                self.optimizer.step()
                if loss.item() < 1e-3:
                    break
            assert not torch.isnan(loss), 'Loss is Nan!'

        # update the design matrix
        self.model.zero_grad()
        if self.image:
            re = self.model(self.last_cxt.unsqueeze(0))
        else:
            re = self.model(self.last_cxt)
        re.backward()
        grad = torch.cat([p.grad.contiguous().view(-1).detach()
                          for p in self.model.parameters()])
        self.Design += grad * grad


'''
Neural UCB
Adapted from the implementation at https://github.com/ZeroWeight/NeuralTS/blob/master/learner_neural.py
'''


class NeuralUCB(_agent):
    def __init__(self,
                 num_arm,  # number of arms
                 dim_context,  # dimension of context feature
                 model,  # Neural network model
                 optimizer,  # optimizer
                 criterion,  # loss function
                 collector,  # context and reward collector
                 nu,  # exploration variance
                 batch_size=None,  # batchsize to update nn
                 image=False,  # image dataset?
                 reduce=None,  # reduce update frequency
                 reg=1.0,  # regularization weight, lambda in original paper
                 device='cpu',
                 name='NeuralUCB'):
        super(NeuralUCB, self).__init__(name)
        self.image = image
        self.reduce = reduce
        self.num_arm = num_arm
        self.dim_context = dim_context

        self.model = model
        if batch_size:
            self.loader = DataLoader(collector, batch_size=batch_size)
            self.batchsize = batch_size
        else:
            self.loader = None
            self.batchsize = None
        self.optimizer = optimizer
        self.criterion = criterion
        self.collector = collector
        self.nu = nu
        self.reg = reg
        self.device = device
        self.step = 0

        self.num_params = get_param_size(model)
        self.clear()

    def clear(self):
        # self.model.init_weights()
        self.collector.clear()
        self.Design = self.reg * \
                      torch.ones(self.num_params, device=self.device)
        self.last_cxt = 0

    def choose_arm(self, context):
        rewards = []
        grad_list = []
        for i in range(self.num_arm):
            self.model.zero_grad()
            if self.image:
                ri = self.model(context[i:i + 1, :, :, :])
            else:
                ri = self.model(context[i])
            ri.backward()

            grad = torch.cat([p.grad.contiguous().view(-1).detach()
                              for p in self.model.parameters()])
            grad_list.append(grad)
            squared_sigma = self.reg * self.nu * grad * grad / self.Design
            sigma = torch.sqrt(torch.sum(squared_sigma))

            sample_r = ri + sigma
            rewards.append(sample_r.item())
        arm_to_pull = np.argmax(rewards)
        # update the design matrix
        self.Design += grad_list[arm_to_pull] * grad_list[arm_to_pull]
        return arm_to_pull

    def receive_reward(self, arm, context, reward):
        self.collector.collect_data(context, arm, reward)
        self.last_cxt = context

    def update_model(self, num_iter):
        self.step += 1
        if self.reduce:
            if self.step % self.reduce != 0:
                return
        for p in self.optimizer.param_groups:
            p['weight_decay'] = self.reg / self.step

        # update using minibatch
        if self.batchsize and self.batchsize < self.step:
            ploader = sample_data(self.loader)
            for i in range(num_iter):
                contexts, rewards = next(ploader)
                contexts = contexts.to(self.device)
                rewards = rewards.to(dtype=torch.float32, device=self.device)
                self.model.zero_grad()
                pred = self.model(contexts).squeeze(dim=1)
                loss = self.criterion(pred, rewards)
                loss.backward()
                self.optimizer.step()
            assert not torch.isnan(loss), 'Loss is Nan!'
        else:
            contexts, rewards = self.collector.fetch_batch()
            contexts = torch.stack(contexts, dim=0).to(self.device)
            rewards = torch.tensor(
                rewards, dtype=torch.float32, device=self.device)
            self.model.train()
            for i in range(num_iter):
                self.model.zero_grad()
                pred = self.model(contexts).squeeze(dim=1)
                loss = self.criterion(pred, rewards)
                loss.backward()
                self.optimizer.step()
                if loss.item() < 1e-3:
                    break
            assert not torch.isnan(loss), 'Loss is Nan!'

'''
Neural LinUCB 
Adapted from the implementation at https://github.com/ZeroWeight/NeuralTS/blob/master/learner_neural.py
'''


class NeuralLinUCB(_agent):
    def __init__(self,
                 num_arm,  # number of arms
                 dim_context,  # dimension of context feature
                 model,  # Neural network model
                 optimizer,  # optimizer
                 criterion,  # loss function
                 collector,  # context and reward collector
                 nu,  # exploration variance
                 batch_size=None,  # batchsize to update nn
                 image=False,  # image dataset?
                 reduce=None,  # reduce update frequency
                 reg=1.0,  # regularization weight, lambda in original paper
                 device='cpu',
                 name='NeuralLinUCB'):
        super(NeuralLinUCB, self).__init__(name)
        self.image = image
        self.reduce = reduce
        self.num_arm = num_arm
        self.dim_context = dim_context

        self.model = model
        if batch_size:
            self.loader = DataLoader(collector, batch_size=batch_size)
            self.batchsize = batch_size
        else:
            self.loader = None
            self.batchsize = None
        self.optimizer = optimizer
        self.criterion = criterion
        self.collector = collector
        self.nu = nu
        self.reg = reg
        self.device = device
        self.step = 0

        self.num_params = get_param_size(model)
        self.clear()

    def clear(self):
        # self.model.init_weights()
        self.collector.clear()
        self.Design = self.reg * \
                      torch.eye(self.model.fc_dim, device=self.device)
        self.last_cxt = 0
        self.last_feature = 0
        self.bb = torch.zeros((self.model.fc_dim,1), device=self.device)

    def UCB(self, A, phi):
        tmp, LU = torch.solve(phi, A)
        return torch.sqrt(torch.mm(torch.transpose(phi, 0, 1), tmp))

    def choose_arm(self, context):
        with torch.no_grad():
            mu, feature = self.model(context)
            ucb = []
            for i in range(self.num_arm):
                ucb_term = self.nu * self.UCB(self.Design, feature[i].view(-1, 1))
                ucb.append(mu[i].item() + ucb_term.item())
            arm_to_pull = np.argmax(ucb)
            selected_feature = feature[arm_to_pull].view(-1, 1)
            self.last_feature = selected_feature
            # update the design matrix
            self.Design += torch.mm(selected_feature, selected_feature.t())
        return arm_to_pull

    def receive_reward(self, arm, context, reward):
        self.bb += reward * self.last_feature
        theta, LU = torch.solve(self.bb, self.Design)
        with torch.no_grad():
            self.model.fc.weight.copy_(theta.view(1, -1))
        self.collector.collect_data(context, arm, reward)
        self.last_cxt = context

    def update_model(self, num_iter):
        self.step += 1
        if self.reduce:
            if self.step % self.reduce != 0:
                return
        for p in self.optimizer.param_groups:
            p['weight_decay'] = self.reg / self.step

        # update using minibatch
        if self.batchsize and self.batchsize < self.step:
            ploader = sample_data(self.loader)
            for i in range(num_iter):
                contexts, rewards = next(ploader)
                contexts = contexts.to(self.device)
                rewards = rewards.to(dtype=torch.float32, device=self.device)
                self.model.zero_grad()
                pred, _ = self.model(contexts)
                pred = pred.squeeze(dim=1)
                loss = self.criterion(pred, rewards)
                loss.backward()
                self.optimizer.step()
            assert not torch.isnan(loss), 'Loss is Nan!'
        else:
            contexts, rewards = self.collector.fetch_batch()
            contexts = torch.stack(contexts, dim=0).to(self.device)
            rewards = torch.tensor(
                rewards, dtype=torch.float32, device=self.device)
            self.model.train()
            for i in range(num_iter):
                self.model.zero_grad()
                pred, _ = self.model(contexts)
                pred = pred.squeeze(dim=1)
                loss = self.criterion(pred, rewards)
                loss.backward()
                self.optimizer.step()
                if loss.item() < 1e-3:
                    break
            assert not torch.isnan(loss), 'Loss is Nan!'

'''
Neural greedy algo and Neural eps-greedy algo
'''


class NeuralEpsGreedy(_agent):
    def __init__(self,
                 num_arm,
                 dim_context,
                 model,
                 optimizer,
                 criterion,
                 collector,
                 eps=0.0,  # 0<=eps<1, eps=0 -> neural greedy, otherwise -> neural eps-greedy
                 batch_size=None,  # batchsize to update nn
                 reduce=None,  # reduce update frequency
                 device='cpu',
                 name='NeuralEpsGreedy'):
        super(NeuralEpsGreedy, self).__init__(name)
        self.num_arm = num_arm
        self.dim_context = dim_context
        self.reduce = reduce

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.collector = collector
        if batch_size:
            self.loader = DataLoader(collector, batch_size=batch_size)
            self.batchsize = batch_size
        else:
            self.loader = None
            self.batchsize = None

        self.eps = eps
        self.step = 0
        self.device = device

    def clear(self):
        # self.model.init_weights()
        self.collector.clear()
        self.step = 0

    def choose_arm(self, context):
        prob = self.eps / np.sqrt(self.step + 1)
        if np.random.uniform() < prob:
            return np.random.randint(self.num_arm)
        else:
            with torch.no_grad():
                pred = self.model(context)
                arm_to_pull = torch.argmax(pred)
            return int(arm_to_pull)

    def receive_reward(self, arm, context, reward):
        self.collector.collect_data(context, arm, reward)

    def update_model(self, num_iter=5):
        self.step += 1
        self.model.train()
        if self.reduce:
            if self.step % self.reduce != 0:
                return

        if self.batchsize and self.batchsize < self.step:
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
            contexts, rewards = self.collector.fetch_batch()
            contexts = torch.stack(contexts, dim=0).to(self.device)
            rewards = torch.tensor(
                rewards, dtype=torch.float32, device=self.device)

            for i in range(num_iter):
                self.model.zero_grad()
                pred = self.model(contexts).squeeze(dim=1)
                loss = self.criterion(pred, rewards)
                loss.backward()
                self.optimizer.step()
            assert not torch.isnan(loss), "Loss is Nan!"


'''
UCB-GLM algorithm 
'''


class UCBGLM(_agent):
    def __init__(self,
                 num_arm,  # number of arms
                 dim_context,  # dimension of context feature
                 model,  # Linear model
                 optimizer,  # optimizer
                 criterion,  # loss function
                 collector,  # context and reward collector
                 nu,  # exploration variance
                 batch_size=None,  # batchsize to update nn
                 image=False,  # image dataset?
                 reduce=None,  # reduce update frequency
                 lr=1.0,    # learning rate
                 reg=1.0,  # regularization weight, lambda in original paper
                 device='cpu',
                 name='NeuralUCB'):
        super(UCBGLM, self).__init__(name)
        self.image = image
        self.reduce = reduce
        self.num_arm = num_arm
        self.dim_context = dim_context
        self.base_lr = lr

        self.model = model
        if batch_size:
            self.loader = DataLoader(collector, batch_size=batch_size)
            self.batchsize = batch_size
        else:
            self.loader = None
            self.batchsize = None
        self.optimizer = optimizer
        self.criterion = criterion
        self.collector = collector
        self.nu = nu
        self.reg = reg
        self.device = device
        self.step = 0

        self.num_params = get_param_size(model)
        self.clear()

    def clear(self):
        # self.model.init_weights()
        self.collector.clear()
        # self.DesignInv = (1 / self.reg) * \
        #                  torch.eye(self.dim_context,
        #                            device=self.device)  # compute its inverse
        self.Design = 0
        self.last_cxt = 0

    def choose_arm(self, context):
        if self.step <= 30:
            return np.random.randint(0, self.num_arm)
        else:
            tol = 1e-12
            if torch.linalg.det(self.DesignInv) < tol:
                cov = self.DesignInv + 0.00001 * \
                      torch.eye(self.dim_context, device=self.device)
            else:
                cov = self.DesignInv
            norms = [context[i].T @ cov @ context[i] for i in range(self.num_arm)]
            model_mean = self.model(context)
            pred = model_mean.view(-1) + torch.tensor(norms, device=self.device)
            arm_to_pull = torch.argmax(pred).item()
            # update the design matrix
            return arm_to_pull

    def receive_reward(self, arm, context, reward):
        self.collector.collect_data(context, arm, reward)
        self.last_cxt = context
        if self.step < 30:
            self.Design += context.view(-1, 1) @ context.view(-1, 1).T
        # update the inverse of the design matrix
        else:
            if self.step == 30:
                self.DesignInv = torch.linalg.inv(self.Design)
            omega = self.DesignInv @ self.last_cxt
            self.DesignInv = self.DesignInv - omega.view(-1, 1) @ omega.view(-1, 1).T / \
                             (1 + torch.dot(omega, self.last_cxt))

    def update_model(self, num_iter):
        self.step += 1
        if self.step < 30:
            return
        if self.reduce:
            if self.step % self.reduce != 0:
                return
        for p in self.optimizer.param_groups:
            p['weight_decay'] = self.reg / self.step

        set_lr(self.optimizer, base=self.base_lr)
        # update using minibatch
        if self.batchsize and self.batchsize < self.step:
            ploader = sample_data(self.loader)
            for i in range(num_iter):
                contexts, rewards = next(ploader)
                contexts = contexts.to(self.device)
                rewards = rewards.to(dtype=torch.float32, device=self.device)
                self.model.zero_grad()
                pred = self.model(contexts).squeeze(dim=1)
                loss = self.criterion(pred, rewards)
                loss.backward()
                self.optimizer.step()
            assert not torch.isnan(loss), 'Loss is Nan!'
        else:
            contexts, rewards = self.collector.fetch_batch()
            contexts = torch.stack(contexts, dim=0).to(self.device)
            rewards = torch.tensor(
                rewards, dtype=torch.float32, device=self.device)
            self.model.train()

            # def loss_closure():
            #     self.model.zero_grad()
            #     pred = self.model(contexts).squeeze(dim=1)
            #     loss = self.criterion(pred, rewards)
            #     loss.backward()
            for i in range(num_iter):
                self.model.zero_grad()
                pred = self.model(contexts).squeeze(dim=1)
                loss = self.criterion(pred, rewards)
                loss.backward()
                self.optimizer.step()
                if i % 50 == 0:
                    set_lr(self.optimizer, 2)

            assert not torch.isnan(loss), 'Loss is Nan!'


def set_lr(optimizer, step=None, base=None):
    if base:
        for p in optimizer.param_groups:
            p['lr'] = base
    else:
        for p in optimizer.param_groups:
            p['lr'] /= step

'''
GLM-TSL
'''

class GLMTSL(_agent):
    def __init__(self,
                 num_arm,  # number of arms
                 dim_context,  # dimension of context feature
                 model,  # Neural network model
                 optimizer,  # optimizer
                 criterion,  # loss function
                 collector,  # context and reward collector
                 nu,  # exploration variance
                 batch_size=None,  # batchsize to update nn
                 tao=None,
                 reduce=None,  # reduce update frequency
                 reg=1.0,  # regularization weight, lambda in original paper
                 device='cpu',
                 name='NeuralTS'):
        super(GLMTSL, self).__init__(name)
        self.tao = dim_context if tao is None else tao

        self.reduce = reduce
        self.num_arm = num_arm
        self.dim_context = dim_context

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        if batch_size:
            self.loader = DataLoader(collector, batch_size=batch_size)
            self.batchsize = batch_size
        else:
            self.loader = None
            self.batchsize = None
        self.collector = collector
        self.nu = nu
        self.reg = reg
        self.device = device
        self.step = 0

        self.num_params = get_param_size(model)
        self.clear()

    def clear(self):
        # self.model.init_weights()
        self.collector.clear()
        self.Design = 0
        self.last_cxt = 0

    def choose_arm(self, context):
        if self.step <= self.tao:
            return self.step % self.num_arm
        else:
            tol = 1e-12
            if torch.linalg.det(self.DesignInv) < tol:
                cov = self.DesignInv + 0.00001 * \
                  torch.eye(self.dim_context, device=self.device)
            else:
                cov = self.DesignInv
            model_theta = self.model.fc.weight.data
            dist = MultivariateNormal(model_theta.view(-1), self.nu ** 2 * cov)
            theta_tilda = dist.sample()

            arm_to_pull = torch.argmax(context @ theta_tilda).item()
        return arm_to_pull

    def receive_reward(self, arm, context, reward):
        self.collector.collect_data(context, arm, reward)
        self.last_cxt = context
        if self.step < self.tao:
            self.Design += 0.25 * context.view(-1, 1) @ context.view(-1, 1).T
        else:
            if self.step == self.tao:
                self.DesignInv = torch.linalg.inv(self.Design)
            omega = self.DesignInv @ self.last_cxt
            self.DesignInv = self.DesignInv - omega.view(-1, 1) @ omega.view(-1, 1).T / \
                             (1 + torch.dot(omega, self.last_cxt))

    def update_model(self, num_iter):
        self.step += 1
        if self.reduce:
            if self.step % self.reduce != 0:
                return

        for p in self.optimizer.param_groups:
            p['weight_decay'] = self.reg / self.step

        # update using minibatch
        if self.batchsize and self.batchsize < self.step:
            ploader = sample_data(self.loader)
            for i in range(num_iter):
                contexts, rewards = next(ploader)
                contexts = contexts.to(self.device)
                rewards = rewards.to(dtype=torch.float32, device=self.device)
                self.model.zero_grad()
                pred = self.model(contexts).squeeze(dim=1)
                loss = self.criterion(pred, rewards)
                loss.backward()
                self.optimizer.step()
            assert not torch.isnan(loss), 'Loss is Nan!'
        else:
            # update using full batch
            contexts, rewards = self.collector.fetch_batch()
            contexts = torch.stack(contexts, dim=0).to(self.device)
            rewards = torch.tensor(
                rewards, dtype=torch.float32, device=self.device)
            self.model.train()
            for i in range(num_iter):
                self.model.zero_grad()
                pred = self.model(contexts).squeeze(dim=1)
                loss = self.criterion(pred, rewards)
                loss.backward()
                self.optimizer.step()
                if loss.item() < 1e-3:
                    break
            assert not torch.isnan(loss), 'Loss is Nan!'

