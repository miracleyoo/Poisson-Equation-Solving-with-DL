# coding: utf-8
# Author: Zhongyang Zhang

import torch
import torch.nn as nn
from .BasicModule import BasicModule

torch.manual_seed(1)


class MiracleNet(BasicModule):
    def __init__(self, opt):
        super(MiracleNet, self).__init__()
        self.model_name = "Miracle_Net"
        self.convs = nn.Sequential(
            nn.Conv2d(2, 128, 3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(3,1,padding=1),
            nn.Conv2d(128, 128, 3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(3, 1, padding=1),
            nn.Conv2d(128, 256, 3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(3, 1, padding=1)
        )
        self.fc = nn.Sequential(
            nn.Linear((9-3*2) * (41-3*2) * 256, opt.LINER_HID_SIZE),
            nn.BatchNorm1d(opt.LINER_HID_SIZE),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(opt.LINER_HID_SIZE, opt.NUM_CLASSES)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x