import torch.nn as nn
import torch


class InstanceMapper(nn.Module):

    def __init__(self, output_size=5, mode='avg', avg_dim=1, reshape=True):
        super(InstanceMapper, self).__init__()
        self.mode = mode
        self.avg_dim = avg_dim
        self.adaptive_average = nn.AdaptiveAvgPool1d(output_size)
        self.reshape = reshape

    def forward(self, x):  # (batch, channel, sequence)
        if self.reshape:
            x = torch.permute(x, (1, 2, 0))
        output = None
        if self.mode == 'avg':
            output = torch.mean(x, self.avg_dim)
        if self.mode == 'adaptive':
            output = self.adaptive_average(x)
        if self.mode == 'identity':
            output = x
        return output
