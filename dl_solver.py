# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import torch
import os
import threading
import numpy as np
from torch.autograd import Variable
from models import miracle_net, miracle_wide_net, miracle_weight_wide_net, miracle_lineconv_net

lock = threading.Lock()


def save_models(opt, net, epoch, train_loss, best_loss, test_loss):
    # Save a temp model
    train_loss = float(train_loss)
    best_loss = float(best_loss)
    test_loss = float(test_loss)
    if opt.SAVE_TEMP_MODEL:
        net.save(epoch, train_loss / opt.NUM_TRAIN, "temp_model.dat")

    # Save the best model
    if test_loss / opt.NUM_TEST < best_loss:
        best_loss = test_loss / opt.NUM_TEST
        net.save(epoch, train_loss / opt.NUM_TRAIN, "best_model.dat")
    return best_loss


class MyThread(threading.Thread):
    def __init__(self, opt, net, epoch, train_loss, best_loss, test_loss):
        threading.Thread.__init__(self)
        self.opt = opt
        self.net = net
        self.epoch = epoch
        self.train_loss = train_loss
        self.best_loss = best_loss
        self.test_loss = test_loss

    def run(self):
        lock.acquire()
        try:
            self.best_loss = save_models(self.opt, self.net, self.epoch, self.train_loss, self.best_loss,
                                         self.test_loss)
        finally:
            lock.release()


class Config(object):
    def __init__(self):
        self.USE_CUDA = torch.cuda.is_available()
        self.NET_SAVE_PATH = "./source/trained_net/"
        self.MODEL = 'MiracleWeightWideNet'
        self.NUM_CHANNEL = 2
        self.PROCESS_ID = 'PADDING_LOSS1-2_WEI4-2-1-1-NEW_GEN-Interval'
        self.LINER_HID_SIZE = 1024
        self.LENGTH = 41
        self.WIDTH = 9
        self.NUM_CLASSES = 369
        self.LEARNING_RATE = 0.001


def dl_init():
    opt = Config()
    try:
        if opt.MODEL == 'MiracleWeightWideNet':
            net = miracle_weight_wide_net.MiracleWeightWideNet(opt)
        elif opt.MODEL == 'MiracleWideNet':
            net = miracle_wide_net.MiracleWideNet(opt)
        elif opt.MODEL == 'MiracleNet':
            net = miracle_net.MiracleNet(opt)
        elif opt.MODEL == 'MiracleLineConvNet':
            net = miracle_lineconv_net.MiracleLineConvNet(opt)
    except KeyError('Your model is not found.'):
        exit(0)
    else:
        print("==> Model initialized successfully.")
    net_save_prefix = opt.NET_SAVE_PATH + opt.MODEL + '_' + opt.PROCESS_ID + '/'
    temp_model_name = net_save_prefix + "best_model.dat"
    if os.path.exists(temp_model_name):
        net, *_ = net.load(temp_model_name)
        print("Load existing model: %s" % temp_model_name)
        if opt.USE_CUDA:
            net.cuda()
            print("==> Using CUDA.")
    else:
        raise FileNotFoundError()
    return opt, net


def border_loss(matrix_a, matrix_b, opt):
    batch = len(matrix_a)
    vec_sim = (torch.sum(torch.pow(matrix_a - matrix_b, 2))).sqrt()
    std = Variable(torch.Tensor(np.zeros([batch, opt.LENGTH, opt.WIDTH])))
    std[:, 0, :] = 1
    std[:, -1, :] = 1
    if opt.USE_CUDA:
        std = std.cuda()
    matrix_a = matrix_a.resize(batch, opt.LENGTH, opt.WIDTH)
    matrix_b = matrix_b.resize(batch, opt.LENGTH, opt.WIDTH)
    a_border = matrix_a * std
    b_border = matrix_b * std
    return vec_sim + 2 * (torch.sum(torch.pow(a_border - b_border, 2))).sqrt()


def online_training(model_input, net, opt, labels):
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.LEARNING_RATE)
    if opt.USE_CUDA:
        inputs = Variable(torch.Tensor(model_input).cuda())
    else:
        inputs = Variable(torch.Tensor(model_input))
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = border_loss(outputs, labels, opt)

    # loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()


def dl_solver(model_input, net, opt):
    net.eval()
    if opt.USE_CUDA:
        inputs = Variable(torch.Tensor(model_input).cuda())
        outputs = net(inputs)
        outputs = outputs.cpu()
    else:
        inputs = Variable(torch.Tensor(model_input))
        outputs = net(inputs)

    outputs = outputs.data.numpy()
    return outputs
