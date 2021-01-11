# '''DenseNet in PyTorch.'''
# import math
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from torch.autograd import Variable
#
#
# class Bottleneck(nn.Module):
#     def __init__(self, in_planes, growth_rate):
#         super(Bottleneck, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(4*growth_rate)
#         self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
#
#     def forward(self, x):
#         out = self.conv1(F.relu(self.bn1(x)))
#         out = self.conv2(F.relu(self.bn2(out)))
#         out = torch.cat([out,x], 1)
#         return out
#
#
# class Transition(nn.Module):
#     def __init__(self, in_planes, out_planes):
#         super(Transition, self).__init__()
#         self.bn = nn.BatchNorm2d(in_planes)
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
#
#     def forward(self, x):
#         out = self.conv(F.relu(self.bn(x)))
#         out = F.avg_pool2d(out, 2)
#         return out
#
#
# class DenseNet(nn.Module):
#     def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
#         super(DenseNet, self).__init__()
#         self.growth_rate = growth_rate
#
#         num_planes = 2*growth_rate
#         self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)
#
#         self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
#         num_planes += nblocks[0]*growth_rate
#         out_planes = int(math.floor(num_planes*reduction))
#         self.trans1 = Transition(num_planes, out_planes)
#         num_planes = out_planes
#
#         self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
#         num_planes += nblocks[1]*growth_rate
#         out_planes = int(math.floor(num_planes*reduction))
#         self.trans2 = Transition(num_planes, out_planes)
#         num_planes = out_planes
#
#         self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
#         num_planes += nblocks[2]*growth_rate
#         out_planes = int(math.floor(num_planes*reduction))
#         self.trans3 = Transition(num_planes, out_planes)
#         num_planes = out_planes
#
#         self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
#         num_planes += nblocks[3]*growth_rate
#
#         self.bn = nn.BatchNorm2d(num_planes)
#         self.linear = nn.Linear(num_planes, num_classes)
#
#     def _make_dense_layers(self, block, in_planes, nblock):
#         layers = []
#         for i in range(nblock):
#             layers.append(block(in_planes, self.growth_rate))
#             in_planes += self.growth_rate
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.trans1(self.dense1(out))
#         out = self.trans2(self.dense2(out))
#         out = self.trans3(self.dense3(out))
#         out = self.dense4(out)
#         out = F.avg_pool2d(F.relu(self.bn(out)), 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out
#
# def DenseNet121():
#     return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)
#
# def DenseNet169():
#     return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)
#
# def DenseNet201():
#     return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)
#
# def DenseNet161():
#     return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)
#
# def densenet_cifar():
#     return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)
#
# def test_densenet():
#     net = densenet_cifar()
#     x = torch.randn(1,3,32,32)
#     y = net(Variable(x))
#     print(y)
#
# # test_densenet()



import torch
import torch.nn as nn



#"""Bottleneck layers. Although each layer only produces k
#output feature-maps, it typically has many more inputs. It
#has been noted in [37, 11] that a 1×1 convolution can be in-
#troduced as bottleneck layer before each 3×3 convolution
#to reduce the number of input feature-maps, and thus to
#improve computational efficiency."""
class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        #"""In  our experiments, we let each 1×1 convolution
        #produce 4k feature-maps."""
        inner_channel = 4 * growth_rate

        #"""We find this design especially effective for DenseNet and
        #we refer to our network with such a bottleneck layer, i.e.,
        #to the BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) version of H ` ,
        #as DenseNet-B."""
        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)

#"""We refer to layers between blocks as transition
#layers, which do convolution and pooling."""
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #"""The transition layers used in our experiments
        #consist of a batch normalization layer and an 1×1
        #convolutional layer followed by a 2×2 average pooling
        #layer""".
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)

#DesneNet-BC
#B stands for bottleneck layer(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
#C stands for compression factor(0<=theta<=1)
class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_class=100):
        super().__init__()
        self.growth_rate = growth_rate

        #"""Before entering the first dense block, a convolution
        #with 16 (or twice the growth rate for DenseNet-BC)
        #output channels is performed on the input images."""
        inner_channels = 2 * growth_rate

        #For convolutional layers with kernel size 3×3, each
        #side of the inputs is zero-padded by one pixel to keep
        #the feature-map size fixed.
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False)

        self.features = nn.Sequential()

        for index in range(len(nblocks) - 1):
            self.features.add_module("dense_block_layer_{}".format(index), self._make_dense_layers(block, inner_channels, nblocks[index]))
            inner_channels += growth_rate * nblocks[index]

            #"""If a dense block contains m feature-maps, we let the
            #following transition layer generate θm output feature-
            #maps, where 0 < θ ≤ 1 is referred to as the compression
            #fac-tor.
            out_channels = int(reduction * inner_channels) # int() will automatic floor the value
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module("dense_block{}".format(len(nblocks) - 1), self._make_dense_layers(block, inner_channels, nblocks[len(nblocks)-1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(inner_channels, num_class)

    def forward(self, x):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)
        return output

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block

def densenet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def densenet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def densenet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def densenet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)
