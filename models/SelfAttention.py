import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=(1, 1))
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=(1, 1))
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=(1, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps( B X c X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
    """
        batch_size, c, width, height = x.size()
        # print(x.size())
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # (b, c, w*h)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)  # (b, c, w*h)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # (b, w*h, w*h)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)  # (b, c, w*h)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, c, width, height)

        out = self.gamma * out + x
        return out
