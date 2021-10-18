"""
Replications of models from Frankle et al. Lottery Ticket Hypothesis
"""

import torch.nn as nn
from utils.builder import get_builder

from args import args


class Conv2(nn.Module):
    def __init__(self):
        super(Conv2, self).__init__()
        builder = get_builder()
        self.relu = nn.ReLU()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 64, first_layer=True),
            self.relu,
            builder.conv3x3(64, 64),
            self.relu,
            nn.MaxPool2d((2, 2)),
        )

        self.linear = nn.Sequential(
            builder.conv1x1(64 * 16 * 16, 256),
            self.relu,
            builder.conv1x1(256, 256),
            self.relu,
            builder.conv1x1(256, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 64 * 16 * 16, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv4(nn.Module):
    def __init__(self):
        super(Conv4, self).__init__()
        builder = get_builder()
        self.relu = nn.ReLU()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 64, first_layer=True),
            self.relu,
            builder.conv3x3(64, 64),
            self.relu,
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(64, 128),
            self.relu,
            builder.conv3x3(128, 128),
            self.relu,
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(32 * 32 * 8, 256),
            self.relu,
            builder.conv1x1(256, 256),
            self.relu,
            builder.conv1x1(256, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 8192, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv6(nn.Module):
    def __init__(self):
        super(Conv6, self).__init__()
        builder = get_builder()
        self.relu = nn.ReLU()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 64, first_layer=True),
            self.relu,
            builder.conv3x3(64, 64),
            self.relu,
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(64, 128),
            self.relu,
            builder.conv3x3(128, 128),
            self.relu,
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(128, 256),
            self.relu,
            builder.conv3x3(256, 256),
            self.relu,
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(256 * 4 * 4, 256),
            self.relu,
            builder.conv1x1(256, 256),
            self.relu,
            builder.conv1x1(256, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 256 * 4 * 4, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv8(nn.Module):
    def __init__(self):
        super(Conv8, self).__init__()
        builder = get_builder()
        self.relu = nn.ReLU()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 64, first_layer=True),
            self.relu,
            builder.conv3x3(64, 64),
            self.relu,
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(64, 128),
            self.relu,
            builder.conv3x3(128, 128),
            self.relu,
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(128, 256),
            self.relu,
            builder.conv3x3(256, 256),
            self.relu,
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(256, 512),
            self.relu,
            builder.conv3x3(512, 512),
            self.relu,
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(512 * 2 * 2, 256),
            self.relu,
            builder.conv1x1(256, 256),
            self.relu,
            builder.conv1x1(256, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 512 * 2 * 2, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        builder = get_builder()
        self.relu = nn.ReLU()
        self.linear = nn.Sequential(
            builder.conv1x1(28 * 28, 300, first_layer=True),
            self.relu,
            builder.conv1x1(300, 100),
            self.relu,
            builder.conv1x1(100, 10),
        )

    def forward(self, x):
        out = x.view(x.size(0), 28 * 28, 1, 1)
        out = self.linear(out)
        return out.squeeze()


def scale(n):
    return int(n * args.width_mult)


class Conv4Wide(nn.Module):
    def __init__(self):
        super(Conv4Wide, self).__init__()
        builder = get_builder()
        self.relu = nn.ReLU()
        self.convs = nn.Sequential(
            builder.conv3x3(3, scale(64), first_layer=True),
            self.relu,
            builder.conv3x3(scale(64), scale(64)),
            self.relu,
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(scale(64), scale(128)),
            self.relu,
            builder.conv3x3(scale(128), scale(128)),
            self.relu,
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(scale(128) * 8 * 8, scale(256)),
            self.relu,
            builder.conv1x1(scale(256), scale(256)),
            self.relu,
            builder.conv1x1(scale(256), 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), scale(128) * 8 * 8, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv6Wide(nn.Module):
    def __init__(self):
        super(Conv6Wide, self).__init__()
        builder = get_builder()
        self.relu = nn.ReLU()
        self.convs = nn.Sequential(
            builder.conv3x3(3, scale(64), first_layer=True),
            self.relu,
            builder.conv3x3(scale(64), scale(64)),
            self.relu,
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(scale(64), scale(128)),
            self.relu,
            builder.conv3x3(scale(128), scale(128)),
            self.relu,
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(scale(128), scale(256)),
            self.relu,
            builder.conv3x3(scale(256), scale(256)),
            self.relu,
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(scale(256) * 4 * 4, scale(256)),
            self.relu,
            builder.conv1x1(scale(256), scale(256)),
            self.relu,
            builder.conv1x1(scale(256), 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), scale(256) * 4 * 4, 1, 1)
        out = self.linear(out)
        return out.squeeze()
