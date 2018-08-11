import torch
from torch.autograd import Variable
from graphviz import Digraph
from torchviz import make_dot
from models import miracle_lineconv_net
from config import Config
opt = Config()

x = Variable(torch.randn(128,2,41,9))#change 12 to the channel number of network input
model = miracle_lineconv_net.MiracleLineConvNet(opt)
y = model(x)
g = make_dot(y)
g.view()