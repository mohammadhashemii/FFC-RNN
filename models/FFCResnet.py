import torch.nn as nn
from FFC import *

__all__ = ['FFCResNet', 'ffc_resnet18', 'ffc_resnet34',
           'ffc_resnet26', 'ffc_resnet50', 'ffc_resnet101',
           'ffc_resnet152', 'ffc_resnet200', 'ffc_resnext50_32x4d',
           'ffc_resnext101_32x8d']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, down_sample=None, groups=1,
                 base_width=64, dilation=1, ratio_gin=0.5, ratio_gout=0.5, lfu=True, use_se=False, norm_layer=None):
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64")

        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")

        width = int(planes * (base_width / 64.)) * groups

        # Both self.conv2 and self.down_sample layers down_sample the input when
        # stride != 1

        self.conv1 = FfcBnAct(in_planes, width, kernel_size=(3, 3), padding=1, stride=stride,
                              ratio_gin=ratio_gin, ratio_gout=ratio_gout, norm_layer=norm_layer,
                              activation_layer=nn.ReLU, enable_lfu=lfu)

        self.conv2 = FfcBnAct(width, planes * self.expansion, kernel_size=(3, 3), padding=1,
                              ratio_gin=ratio_gout, ratio_gout=ratio_gout, norm_layer=norm_layer, enable_lfu=lfu)

        self.se_block = FfcseBlock(
            planes * self.expansion, ratio_gout) if use_se else nn.Identity()

        self.relu_l = nn.Identity() if ratio_gout == 1 else nn.ReLU(inplace=True)
        self.relu_g = nn.Identity() if ratio_gout == 0 else nn.ReLU(inplace=True)

        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x if self.down_sample is None else self.down_sample(x)

        x = self.conv1(x)
        # x_l, x_g = self.conv2(x)
        x_l, x_g = self.se_block(x)

        x_l = self.relu_l(x_l + id_l)
        x_g = self.relu_g(x_g + id_g)

        return x_l, x_g


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, down_sample=None, groups=1,
                 base_width=64, dilation=1, ratio_gin=0.5, ratio_gout=0.5, lfu=True, use_se=False):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.down_sample layers down_sample the input when
        # stride != 1

        self.conv1 = FfcBnAct(in_planes, width, kernel_size=(1, 1),
                              ratio_gin=ratio_gin, ratio_gout=ratio_gout,
                              activation_layer=nn.ReLU, enable_lfu=lfu)

        self.conv2 = FfcBnAct(width, width, kernel_size=(3, 3),
                              ratio_gin=ratio_gout, ratio_gout=ratio_gout,
                              stride=stride, padding=1, groups=groups,
                              activation_layer=nn.ReLU, enable_lfu=lfu)

        self.conv3 = FfcBnAct(width, planes * self.expansion, kernel_size=(1, 1),
                              ratio_gin=ratio_gout, ratio_gout=ratio_gout, enable_lfu=lfu)

        self.se_block = FfcseBlock(
            planes * self.expansion, ratio_gout) if use_se else nn.Identity()

        self.relu_l = nn.Identity() if ratio_gout == 1 else nn.ReLU(inplace=True)
        self.relu_g = nn.Identity() if ratio_gout == 0 else nn.ReLU(inplace=True)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x if self.down_sample is None else self.down_sample(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_l, x_g = self.se_block(x)

        x_l = self.relu_l(x_l + id_l)
        x_g = self.relu_g(x_g + id_g)

        return x_l, x_g


class FFCResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None, ratio=0.5, lfu=True, use_se=False):
        super(FFCResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        in_planes = 64
        # TODO add ratio-in_planes-groups assertion

        self.in_planes = in_planes
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.lfu = lfu
        self.use_se = use_se
        # self.conv1 = nn.Conv2d(3, in_planes, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.conv1 = nn.Conv2d(1, in_planes, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.bn1 = norm_layer(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, in_planes * 1, layers[0], stride=1, ratio_gin=0, ratio_gout=ratio)
        self.layer2 = self._make_layer(block, in_planes * 2, layers[1], stride=2, ratio_gin=ratio, ratio_gout=ratio)
        self.layer3 = self._make_layer(block, in_planes * 4, layers[2], stride=2, ratio_gin=ratio, ratio_gout=ratio)
        self.layer4 = self._make_layer(block, in_planes * 8, layers[3], stride=2, ratio_gin=ratio, ratio_gout=0)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_planes * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, ratio_gin=0.5, ratio_gout=0.5):
        norm_layer = self._norm_layer
        down_sample = None

        if stride != 1 or self.in_planes != planes * block.expansion or ratio_gin == 0:
            down_sample = FfcBnAct(self.in_planes, planes * block.expansion, kernel_size=(1, 1), stride=stride,
                                   ratio_gin=ratio_gin, ratio_gout=ratio_gout, enable_lfu=self.lfu)

        layers = [block(self.in_planes, planes, stride, down_sample, self.groups, self.base_width,
                        self.dilation, ratio_gin, ratio_gout, lfu=self.lfu, use_se=self.use_se)]

        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(self.in_planes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation,
                      ratio_gin=ratio_gout, ratio_gout=ratio_gout, lfu=self.lfu, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        print("input size in ffcResnet", x.size())

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        print(x.size())

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        print("after layers size in ffcResnet", x[0].size())
        print(x[1])

        x = self.avg_pool(x[0])
        x = x.view(x.size(0), -1)

        print("after layers 2 size in ffcResnet", x[0].size())
        # print(x[1])

        x = self.fc(x)

        return x


def ffc_resnet18(pretrained=False, **kwargs):
    """Constructs a FFT ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = FFCResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def ffc_resnet34(pretrained=False, **kwargs):
    """Constructs a FFT ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = FFCResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def ffc_resnet26(pretrained=False, **kwargs):
    """Constructs a FFT ResNet-26 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FFCResNet(Bottleneck, [2, 2, 2, 2], **kwargs)
    return model


def ffc_resnet50(pretrained=False, **kwargs):
    """Constructs a FFT ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = FFCResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def ffc_resnet101(pretrained=False, **kwargs):
    """Constructs a FFT ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = FFCResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def ffc_resnet152(pretrained=False, **kwargs):
    """Constructs a FFT ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = FFCResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def ffc_resnet200(pretrained=False, **kwargs):
    """Constructs a FFT ResNet-200 model.
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """
    model = FFCResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


def ffc_resnext50_32x4d(pretrained=False, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    model = FFCResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def ffc_resnext101_32x8d(pretrained=False, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    model = FFCResNet(Bottleneck, [3, 4, 32, 3], **kwargs)

    return model


if __name__ == '__main__':
    model = ffc_resnet18()
    # model = ffc_resnet26()
    print("++++++++++++++++++++++++++")
    # tensor = torch.zeros([10, 1, 256, 256], dtype=torch.float32)
    tensor = torch.zeros([10, 1, 32, 256], dtype=torch.float32)
    res = model(tensor)

    # x = torch.zeros([10, 16, 64, 64], dtype=torch.float32)
    # x = torch.zeros([10, 16, 8, 64], dtype=torch.float32)
    # n, c, h, w = x.shape
    # print("lfu x size ", x.shape)
    # split_no = 2
    # split_s = h // split_no
    #
    # split = torch.split(x[:, :c // 4], split_s, dim=-2)
    # print("xs size after split", split[0].size())
    # print("xs size after split", split[1].size())
    # xs = torch.cat(split, dim=1).contiguous()
    # print("xs size after concat", xs.shape)
    # xs = torch.cat(torch.split(xs, (w // split_no), dim=-1), dim=1).contiguous()
    # print("xs final size", xs.size())
