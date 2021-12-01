import torch.nn as nn


class LinearNet(nn.Module):
    def __init__(self, num_arms, dim_context, norm=False):
        super(LinearNet, self).__init__()
        self.net = nn.Linear(dim_context, num_arms, bias=False)

    def forward(self, x):
        '''
        Input: 
            - x: context vector with dim_context dimensions, (N, dim_context)
        Output: 
            - output: predicted reward for each arm, (N, num_arms)
        '''
        output = self.net(x)
        return output

    def init_weights(self):

        self.net.reset_parameters()
