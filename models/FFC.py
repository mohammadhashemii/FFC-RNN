import torch
import torch.nn as nn


# import FFCResnet

class FfcseBlock(nn.Module):

    def __init__(self, channels, ratio_g):
        super(FfcseBlock, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r, kernel_size=(1, 1), bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(channels // r, in_cl, kernel_size=(1, 1), bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(channels // r, in_cg, kernel_size=(1, 1), bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avg_pool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * self.sigmoid(self.conv_a2g(x))
        return x_l, x_g


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=(1, 1), stride=(1, 1), padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        # print(x.size())
        # fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        # print(ffted.size())

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4,
                                                                       2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        # ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim)

        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu

        if stride == 2:
            self.down_sample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.down_sample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(1, 1), groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )

        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups)

        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)

        self.conv2 = torch.nn.Conv2d(out_channels // 2, out_channels, kernel_size=(1, 1), groups=groups, bias=False)

    def forward(self, x):
        # print("input x size at SpectralTransform", x.size())
        x = self.down_sample(x)
        x = self.conv1(x)

        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            # print("lfu x size at SpectralTransform", x.shape)
            split_no = 2
            split_s = h // split_no
            split_s_w = w // split_no
            xs = torch.cat(torch.split(x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            # xs = torch.cat(torch.split(xs, split_s, dim=-1), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1), dim=1).contiguous()
            # print("xs size at SpectralTransform", xs.size())
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        self.in_cg = in_cg
        self.in_cl = in_cl

        # print("in_cg", in_cg)
        # print("in_cl", in_cl)
        # print("out_cg", out_cg)
        # print("out_cl", out_cl)
        # groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        # groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.conv_l2l = module(in_cl, out_cl, kernel_size, stride, padding, dilation, groups, bias)

        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.conv_l2g = module(in_cl, out_cg, kernel_size, stride, padding, dilation, groups, bias)

        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.conv_g2l = module(in_cg, out_cl, kernel_size, stride, padding, dilation, groups, bias)

        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.conv_g2g = module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)
        # self.conv_g2g = module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, False)

    def forward(self, x):
        # x_l, x_g = x if type(x) is tuple else (x, 0)
        x_l, x_g = x if type(x) is tuple else (x[:, :self.in_cl, ...], x[:, self.in_cl:, ...])
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl = self.conv_l2l(x_l) + self.conv_g2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.conv_l2g(x_l) + self.conv_g2g(x_g)

        return out_xl, out_xg


class FfcBnAct(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 enable_lfu=True, merge=False):
        super(FfcBnAct, self).__init__()

        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu)

        l_norm = nn.Identity if ratio_gout == 1 else norm_layer
        g_norm = nn.Identity if ratio_gout == 0 else norm_layer

        self.bn_l = l_norm(int(out_channels * (1 - ratio_gout)))
        self.bn_g = g_norm(int(out_channels * ratio_gout))

        l_act = nn.Identity if ratio_gout == 1 else activation_layer
        g_act = nn.Identity if ratio_gout == 0 else activation_layer

        self.act_l = l_act(inplace=True)
        self.act_g = g_act(inplace=True)
        self.merge = merge

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))

        if self.merge:
            return torch.cat((x_l, x_g), dim=1)
        else:
            return x_l, x_g
