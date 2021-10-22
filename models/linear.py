import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(in_channel, out_channel, bias=False)
    
    def forward(self, x):
        return self.fc(x)
    
    def init_weights(self):
        nn.init.xavier_normal_(self.fc.weight)