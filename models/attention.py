import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn import Conv2d
try:
    from .prototype import NN
except:
    from prototype import NN


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class wide_basic(NN):

    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = Conv2d(in_planes, planes, (3, 3), 1, 1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, (3, 3), stride, 1)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, planes, stride),
            )

    def weight_initialization(self):
        self.conv1.weight.data = torch.FloatTensor(NN.weight_relu_initialization(self.conv1))
        self.conv1.bias.data = torch.FloatTensor(NN.bias_initialization(self.conv1, constant=0))
        self.conv2.weight.data = torch.FloatTensor(NN.weight_relu_initialization(self.conv2))
        self.conv2.bias.data = torch.FloatTensor(NN.bias_initialization(self.conv2, constant=0))

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        # accumulate output

        return out


class Attention(NN):

    def __init__(self, depth, widen_factor, dropout_rate, num_classes, base=16):
        super(Attention, self).__init__()
        self.in_planes = base

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor

        print('| Wide-Resnet %dx%d' % (depth, k))
        nStages = [base, base * k, base * k * 2, base * k * 4]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = conv1x1(nStages[3], num_classes)
        self.name = 'Wide_ResNet_{}_{}_{}_{}'.format(depth, widen_factor, dropout_rate, num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * int(num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        [layer.weight_initialization() for layer in layers]
        return nn.Sequential(*layers)

    def weight_initialization(self):
        self.conv1.weight.data = torch.FloatTensor(NN.weight_relu_initialization(self.conv1))
        self.conv1.bias.data = torch.FloatTensor(NN.bias_initialization(self.conv1, constant=0))
        self.linear.weight.data = torch.FloatTensor(np.random.uniform(-0.1, 0.1, self.linear.weight.data.shape))
        self.linear.bias.data = torch.FloatTensor(NN.bias_initialization(self.linear, constant=0.0))

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = self.linear(out)
        # prediction
        prediction = F.avg_pool2d(out, 8)
        prediction = prediction.view(out.size(0), -1)
        # attention map
        attention = F.softmax(prediction).view(prediction.size(0), prediction.size(1), 1, 1).repeat(1, 1, out.size(2), out.size(3)) * out
        attention = F.upsample_bilinear(attention.sum(1).view(attention.size(0), 1, attention.size(2), attention.size(3)).repeat(1, 3, 1, 1), x.size()[2:])
        return prediction, attention


if __name__ == '__main__':
    net = Attention(16, 1, 0.3, 10)
    y = net(Variable(torch.randn(1, 3, 32, 32)))
    print(y.size())
