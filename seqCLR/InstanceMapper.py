import torch.nn as nn
import torch


class InstanceMapper(nn.Module):

    def __init__(self, output_size, sequence_length=32, mode='avg', avg_dim=1, reshape=True):
        super(InstanceMapper, self).__init__()
        self.mode = mode
        self.avg_dim = avg_dim
        self.adaptive_average = nn.AdaptiveAvgPool1d(output_size)
        self.reshape = reshape
        self.output_size = output_size
        self.sequence_length = sequence_length

    def get_output_size(self):
        if self.mode == 'avg':
            return 1
        if self.mode == 'identity':
            return self.sequence_length
        return self.output_size

    def forward(self, x):  # (batch, channel, sequence)
        if self.reshape:
            x = torch.permute(x, (1, 2, 0))
        output = None
        if self.mode == 'avg':
            output = torch.mean(x, self.avg_dim)
        if self.mode == 'adaptive':
            x = torch.split(self.adaptive_average(x), dim=2, split_size_or_sections=1)
            output = torch.squeeze(torch.cat(x, dim=0))
        if self.mode == 'identity':
            x = torch.split(x, dim=2, split_size_or_sections=1)
            output = torch.squeeze(torch.cat(x, dim=0))
        return output

