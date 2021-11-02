import torch
import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(in_channel, out_channel, bias=False)

    def forward(self, x):
        norm_factor = torch.norm(x, dim=1, keepdim=True)
        x = x / norm_factor
        return self.fc(x)

    def init_weights(self):
        self.fc.reset_parameters()
