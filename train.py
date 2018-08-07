# coding: utf-8
import torch
import torch.nn as nn
import torch.autograd
import os
import math
import json
import datetime
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm


def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


def vec_similarity(A, B):
    return (torch.sum(torch.pow(A-B, 2))).sqrt()


def border_loss(A, B, opt, is_training=True):
    batch = len(A)
    vec_sim = (torch.sum(torch.pow(A-B, 2))).sqrt()
    std = Variable(torch.Tensor(np.zeros([batch, opt.LENGTH,opt.WIDTH])))
    std[:, 0, :] = 1
    std[:, -1, :] = 1
    A = A.resize(batch, opt.LENGTH, opt.WIDTH)
    B = B.resize(batch, opt.LENGTH, opt.WIDTH)
    A_bor = A*std
    B_bor = B*std
    return vec_sim + 2*(torch.sum(torch.pow(A_bor-B_bor, 2))).sqrt()


def training(opt, train_loader, test_loader, net):
    NUM_TRAIN_PER_EPOCH = len(train_loader)

    print('==> Loading Model ...')

    temp_model_name = opt.NET_SAVE_PATH +  '%s_model_temp.pkl' % net.__class__.__name__
    if os.path.exists(temp_model_name) and not opt.RE_TRAIN:
        net = torch.load(temp_model_name)
        print("Load existing model: %s" % temp_model_name)

    if opt.USE_CUDA: net.cuda();print("==> Using CUDA.")

    optimizer = torch.optim.Adam(net.parameters(), lr=opt.LEARNING_RATE)

    for epoch in range(opt.NUM_EPOCHS):
        train_loss = 0

        # Start training
        net.train()

        print('==> Preparing Data ...')
        for i, data in tqdm(enumerate(train_loader), desc="Training", total=NUM_TRAIN_PER_EPOCH, leave=False, unit='b'):
            inputs, labels, *_ = data
            if opt.USE_CUDA:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = border_loss(outputs, labels, opt, is_training=True)

            # loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Do statistics for training
            train_loss += loss.data[0]

        # Save a temp model
        torch.save(net, temp_model_name)

        # Start testing
        test_loss = testing(opt, test_loader, net)

        # Output results
        print(
            'Epoch [%d/%d], Train Loss: %.4f Test Loss: %.4f'
            % (epoch + 1, opt.NUM_EPOCHS, train_loss / opt.NUM_TRAIN,
               test_loss / opt.NUM_TEST, ))

    print('==> Training Finished.')
    return net


def testing(opt, test_loader, net):
    net.eval()
    test_loss = 0

    for i, data in tqdm(enumerate(test_loader), desc="Testing", total=len(test_loader), leave=False, unit='b'):
        inputs, labels, *_ = data
        if opt.USE_CUDA:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # Compute the outputs and judge correct
        outputs = net(inputs)
        loss = border_loss(outputs, labels, opt, is_training=False)
        test_loss += loss.data[0]

    return test_loss


def output_vector(opt, net, data):
    net.eval()
    inputs, labels, *_ = data
    if opt.USE_CUDA:
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)
    outputs = net(inputs)
    return outputs
