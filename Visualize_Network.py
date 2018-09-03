# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import torch
from torch.autograd import Variable
from torchviz import make_dot
from models import miracle_lineconv_net
from config import Config

opt = Config()

x = Variable(torch.randn(128, 2, 41, 9))
model = miracle_lineconv_net.MiracleLineConvNet(opt)
y = model(x)
g = make_dot(y)
g.view()
