# coding: utf-8
# Author: Zhongyang Zhang

import torch
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import miracle_net, miracle_wide_net, miracle_weight_wide_net, miracle_lineconv_net
from torch.utils.data import Dataset

#
# class POISSON(Dataset):
#     def __init__(self, data):
#         super(POISSON, self).__init__()
#         self.data = data
#
#     def __len__(self):
#         return 1
#
#     def __getitem__(self, index):
#         inputs = self.data[index]
#         return torch.from_numpy(inputs).float()


class Config(object):
    def __init__(self):
        self.USE_CUDA            = torch.cuda.is_available()
        self.NET_SAVE_PATH       = "./source/trained_net/"
        self.MODEL               = 'MiracleWeightWideNet'
        self.PROCESS_ID          = 'PADDING_LOSS1-2_WEI4-2-2-FULL_SET'
        # self.NUM_TEST            = 0
        # self.NUM_TRAIN           = 0
        # self.NUM_CLASSES         = 369
        # self.LEARNING_RATE       = 0.001
        # self.LINER_HID_SIZE      = 1024
        # self.LENGTH              = 41
        # self.WIDTH               = 9
        # self.TEST_BATCH_SIZE     = 1
        # self.NUM_WORKERS         = 1


def dl_init():
    opt = Config()
    if opt.MODEL == 'MiracleWeightWideNet':
        net = miracle_weight_wide_net.MiracleWeightWideNet(opt)
    elif opt.MODEL == 'MiracleWideNet':
        net = miracle_wide_net.MiracleWideNet(opt)
    elif opt.MODEL == 'MiracleNet':
        net = miracle_net.MiracleNet(opt)
    elif opt.MODEL == 'MiracleLineConvNet':
        net = miracle_lineconv_net.MiracleLineConvNet(opt)

    NET_SAVE_PREFIX = opt.NET_SAVE_PATH + opt.MODEL + '_' + opt.PROCESS_ID + '/'
    temp_model_name = NET_SAVE_PREFIX + "best_model.dat"
    if os.path.exists(temp_model_name):
        net, *_ = net.load(temp_model_name)
        print("Load existing model: %s" % temp_model_name)
    else:
        FileNotFoundError()
    return opt, net


def dl_solver(model_input, net, opt):
    # testDataset = POISSON(model_input)
    # test_loader = DataLoader(dataset=testDataset, batch_size=1, num_workers=1)
    # for i, data in enumerate(test_loader):
    # inputs, *_ = data
    net.eval()
    if opt.USE_CUDA:
        inputs = Variable(torch.Tensor(model_input).cuda())
    else:
        inputs = Variable(torch.Tensor(model_input))
    outputs = net(inputs)
    outputs = outputs.data.numpy()
    return outputs
