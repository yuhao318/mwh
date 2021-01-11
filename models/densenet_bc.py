# import math
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
#
#
# def flatten(x):
#     return x.view(x.size(0), -1)
#
# class SingleLayer(nn.Module):
#     '''
#     Before entering the first dense vlock, a convolution with 16 (or twice the
#     growth rate for BC type) output channels is performed on the input images
#     '''
#     def __init__(self, nChannels, growthRate):
#         super(SingleLayer, self).__init__()
#
#         self.bn1 = nn.BatchNorm2d(nChannels)
#         self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         out = self.conv1(self.relu(self.bn1(x)))
#         out = torch.cat((x, out), 1)
#         return out
#
# class Bottleneck(nn.Module):
#
#     def __init__(self, nChannels, growthRate):
#         super(Bottleneck, self).__init__()
#
#         self.bn1 = nn.BatchNorm2d(nChannels)
#         self.conv1 = nn.Conv2d(nChannels, 4*growthRate, kernel_size=1, bias=False)
#
#         self.bn2 = nn.BatchNorm2d(4*growthRate)
#         self.conv2 = nn.Conv2d(4*growthRate, growthRate, kernel_size=3, padding=1, bias=False)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         out = self.conv1(self.relu(self.bn1(x)))
#         out = self.conv2(self.relu(self.bn2(out)))
#         out = torch.cat((x, out), 1)
#         return out
#
#
# class Transition(nn.Module):
#
#     def __init__(self, nChannels, nOutChannels):
#         super(Transition, self).__init__()
#
#         self.bn1 = nn.BatchNorm2d(nChannels)
#         self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)
#         self.avgpool = nn.AvgPool2d(kernel_size=2)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         out = self.conv1(self.relu(self.bn1(x)))
#         out = self.avgpool(out)
#         return out
#
#
# class DenseNet(nn.Module):
#
#     def __init__(self, name, growthRate, depth, reduction, nClasses, bottleneck):
#         super(DenseNet, self).__init__()
#
#         self.name = name
#         compression = True if reduction < 1 else False  # Determine if DenseNet-C
#
#         nDenseLayers = (depth-4) // 3
#         if bottleneck:
#             nDenseLayers //= 2
#
#         nChannels = 2 * growthRate if compression and bottleneck else 16
#
#         # First convolution
#         self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias=False)
#
#         # Dense Block 1
#         self.dense1 = self._make_dense(nChannels, growthRate, nDenseLayers, bottleneck)
#         nChannels += nDenseLayers*growthRate
#         nOutChannels = int(math.floor(nChannels*reduction))
#
#         # Transition Block 1
#         self.trans1 = Transition(nChannels, nOutChannels)
#         nChannels = nOutChannels
#
#         # Dense Block 2
#         self.dense2 = self._make_dense(nChannels, growthRate, nDenseLayers, bottleneck)
#         nChannels += nDenseLayers*growthRate
#         nOutChannels = int(math.floor(nChannels*reduction))
#
#         # Transition Block 2
#         self.trans2 = Transition(nChannels, nOutChannels)
#         nChannels = nOutChannels
#
#         # Dense Block 3
#         self.dense3 = self._make_dense(nChannels, growthRate, nDenseLayers, bottleneck)
#
#         # Transition Block 3
#         nChannels += nDenseLayers*growthRate
#         self.bn1 = nn.BatchNorm2d(nChannels)
#
#         # Dense Layer
#         self.avgpool = nn.AvgPool2d(kernel_size=8)
#
#         self.flatten = flatten
#         self.fc = nn.Linear(nChannels, nClasses)
#
#         # Initialization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.bias.data.zero_()
#
#     def _make_dense(self, nChannels, growthRate, nDenseLayers, bottleneck):
#         ''' Function to build a Dense Block '''
#         layers = []
#         for i in range(int(nDenseLayers)):
#             if bottleneck:
#                 layers.append(Bottleneck(nChannels, growthRate))
#             else:
#                 layers.append(SingleLayer(nChannels, growthRate))
#             nChannels += growthRate
#         return nn.Sequential(*layers)
#
#
#     def forward(self, x):
#         out = self.conv1(x)                     # 32x32
#         out = self.trans1(self.dense1(out))     # 16x16
#         out = self.trans2(self.dense2(out))     # 8x8
#         out = self.dense3(out)                  # 8x8
#         out = self.avgpool(out)                 # 1x1
#         out = self.flatten(out)
#         out = self.fc(out)
#         return out
#
#
#
# def denseNet_40_12():
#     return DenseNet('DenseNet_12_40', 12, 40, 1, 10, bottleneck=False)
#
# def denseNet_100_12():
#     return DenseNet('DenseNet_12_100', 12, 100, 1, 10, bottleneck=False)
#
# def denseNet_100_24():
#     return DenseNet('DenseNet_24_100', 24, 100, 1, 10, bottleneck=False)
#
#
# def denseNetBC_100_12():
#     return DenseNet('DenseNet-BC_12_100', 12, 100, 0.5, 10, bottleneck=True)
#
# def denseNetBC_250_24():
#     return DenseNet('DenseNet-BC_24_250', 24, 250, 0.5, 10, bottleneck=True)
#
# def denseNetBC_190_40():
#     return DenseNet('DenseNet-BC_40_190', 40, 190, 0.5, 10, bottleneck=True)
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import math


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet_Cifar(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=12, block_config=(16, 16, 16),
                 num_init_features=24, bn_size=4, drop_rate=0, num_classes=10):

        super(DenseNet_Cifar, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # initialize conv and bn parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=8, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out

#
# def densenet_BC_cifar(depth, k, **kwargs):
#     N = (depth - 4) // 6
#     model = DenseNet_Cifar(growth_rate=k, block_config=[N, N, N], num_init_features=2 * k, **kwargs)
#     return model

def denseNetBC_190_40():
    depth = 190
    k = 40
    N = (depth - 4) // 6
    model = DenseNet_Cifar(growth_rate=k, block_config=[N, N, N], num_init_features=2 * k)

    return model
