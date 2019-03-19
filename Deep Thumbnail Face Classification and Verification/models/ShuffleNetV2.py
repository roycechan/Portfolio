import torch
from torch import nn
from torch.autograd import Variable

import config

def init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def conv(in_channels, out_channels, kernel_size, stride):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))
    return conv


def channel_shuffle(x, num_groups):
    N, C, H, W = x.size()
    x_reshape = x.reshape(N, num_groups, C // num_groups, H, W)
    x_permute = x_reshape.permute(0, 2, 1, 3, 4)
    return x_permute.reshape(N, C, H, W)


class BasicUnit(nn.Module):
    def __init__(self, in_channels, splits=2, groups=2):
        super(BasicUnit, self).__init__()
        self.in_channels = in_channels
        self.splits = splits
        self.groups = groups

        in_channels = int(in_channels / self.splits)
        self.right = nn.Sequential(*[
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        ])

        init_weights(self)

    def forward(self, x):
        split = torch.split(x, int(self.in_channels / self.splits), dim=1)
        x_left, x_right = split
        x_right = self.right(x_right)
        x = torch.cat([x_left, x_right], dim=1)
        out = channel_shuffle(x, self.groups)
        # print("Basic Unit", out.size())
        return out


class DownUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=2):
        super(DownUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        self.left = nn.Sequential(*[
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=2, bias=False, groups=self.in_channels),
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(self.in_channels, self.out_channels // 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.out_channels // 2),
            nn.ReLU(inplace=True)
        ])
        self.right = nn.Sequential(*[
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=2, bias=False, groups=self.in_channels),
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(self.in_channels, self.out_channels // 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.out_channels // 2),
            nn.ReLU(inplace=True)
        ])

        init_weights(self)

    def forward(self, x):
        x_left = self.left(x)
        x_right = self.right(x)
        x = torch.cat([x_left, x_right], dim=1)
        out = channel_shuffle(x, self.groups)
        # print("Down Unit", out.size())
        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, n_class, net_size):
        super(ShuffleNetV2, self).__init__()
        out_channels = config.net_size[net_size]
        num_blocks = config.net_blocks
        self.conv1 = conv(in_channels=3, out_channels=out_channels[0],
                          kernel_size=config.conv1_kernel_size,
                          stride=config.conv1_stride)
        self.in_channels = out_channels[0]
        self.stage2 = self._make_stage(out_channels[1], num_blocks[0])
        self.stage3 = self._make_stage(out_channels[2], num_blocks[1])
        # self.stage4 = self._make_stage(out_channels[3], num_blocks[2])
        self.conv5 = conv(in_channels=out_channels[2],
                          out_channels=out_channels[3],
                          kernel_size=config.conv5_kernel_size,
                          stride=config.conv5_stride)
        self.global_pool = nn.AvgPool2d(kernel_size=config.global_pool_kernel_size)
        self.fc = nn.Linear(out_channels[3], n_class)

    def _make_stage(self, out_channels, num_blocks):
        stage = []
        stage.append(DownUnit(self.in_channels, out_channels))
        for i in range(num_blocks):
            stage.append(BasicUnit(out_channels))
        self.in_channels = out_channels  # update in_channels for next iter
        return nn.Sequential(*stage)

    def forward(self, x):
        out = self.conv1(x)
        out = self.stage2(out)
        out = self.stage3(out)
        # out = self.stage4(out)
        out = self.conv5(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)  # flatten
        out = self.fc(out)
        return out


def test():
    net = ShuffleNetV2(2300, 2)
    x = Variable(torch.randn(3, 3, 32, 32))
    y = net(x)
    print("end", y.size())


if __name__ == '__main__':
    test()
