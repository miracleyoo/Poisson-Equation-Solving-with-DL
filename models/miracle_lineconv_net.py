# coding: utf-8
# Author: Zhongyang Zhang

import torch
import torch.nn as nn
from .BasicModule import BasicModule

torch.manual_seed(1)


class MeanRepeat(nn.Module):
    def __init__(self):
        super(MeanRepeat, self).__init__()

    def forward(self, x):
        x = x.mean(2).unsqueeze_(2).repeat(1, 1, 9, 1)
        return x


class MiracleLineConvNet(BasicModule):
    def __init__(self, opt):
        super(MiracleLineConvNet, self).__init__(opt)
        self.model_name = "Miracle_Line_Conv_Net"
        self.weight = {3: 4, 5: 2, 7: 1, 9: 1}
        self.smooth = MeanRepeat()
        init_convs = [nn.Sequential(
            nn.Conv2d(in_channels=opt.NUM_CHANNEL,
                      out_channels=16 * self.weight[kernel_size],
                      kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(16 * self.weight[kernel_size]),
            nn.LeakyReLU(),
            nn.MaxPool2d(3, 1, padding=1)
        ) for kernel_size in (3, 5)]

        self.lineconv = nn.Sequential(
            self.smooth,
            nn.Conv2d(in_channels=opt.NUM_CHANNEL, out_channels=32,
                      kernel_size=3, padding=1),
            # nn.Conv2d(in_channels=opt.NUM_CHANNEL, out_channels=32,
            #           kernel_size=(1, 9), padding=(0, 4)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(3, 1, padding=1)
        )
        init_convs.append(self.lineconv)
        self.init_convs = nn.ModuleList(init_convs)

        self.convs = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear((9 - 4 * 2 * 0) * (41 - 4 * 2 * 0) * 512, opt.LINER_HID_SIZE),
            nn.BatchNorm1d(opt.LINER_HID_SIZE),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(opt.LINER_HID_SIZE, opt.NUM_CLASSES)
        )

    def forward(self, x):
        x = [init_conv(x) for init_conv in self.init_convs]
        x = torch.cat(x, dim=1)
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
