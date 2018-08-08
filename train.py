# coding: utf-8
# Author: Zhongyang Zhang

import torch
import torch.nn as nn
import torch.autograd
import os
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter


def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


def vec_similarity(A, B):
    return (torch.sum(torch.pow(A - B, 2))).sqrt()


def vec_dif(A, B):
    print("Different between preds and labels is:", torch.mean(torch.abs(A-B)).data.tolist())
    print('')


def border_loss(A, B, opt):
    batch = len(A)
    vec_sim = (torch.sum(torch.pow(A - B, 2))).sqrt()
    std = Variable(torch.Tensor(np.zeros([batch, opt.LENGTH, opt.WIDTH])))
    std[:, 0, :] = 1
    std[:, -1, :] = 1
    if opt.USE_CUDA: std = std.cuda()
    A = A.resize(batch, opt.LENGTH, opt.WIDTH)
    B = B.resize(batch, opt.LENGTH, opt.WIDTH)
    A_bor = A * std
    B_bor = B * std
    return vec_sim + 2 * (torch.sum(torch.pow(A_bor - B_bor, 2))).sqrt()


def training(opt, train_loader, test_loader, net):
    NUM_TRAIN_PER_EPOCH = len(train_loader)
    best_loss = 100
    PRE_EPOCH = 0
    print('==> Loading Model ...')

    NET_SAVE_PREFIX = "./source/trained_net/" + net.model_name
    temp_model_name = NET_SAVE_PREFIX + "/temp_model.dat"
    # temp_model_name = opt.NET_SAVE_PATH +  '%s_model_temp.pkl' % net.__class__.__name__
    # best_model_name = opt.NET_SAVE_PATH + '%s_model_best.pkl' % net.__class__.__name__
    if not os.path.exists(NET_SAVE_PREFIX):
        os.mkdir(NET_SAVE_PREFIX)
    if os.path.exists(temp_model_name) and opt.LOAD_SAVED_MOD:
        # net = torch.load(temp_model_name)
        net, PRE_EPOCH, best_loss = net.load(temp_model_name)
        print("Load existing model: %s" % temp_model_name)

    if opt.USE_CUDA:
        net.cuda()
        print("==> Using CUDA.")

    writer = SummaryWriter(opt.SUMMARY_PATH)

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
            loss = border_loss(outputs, labels, opt)

            # loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Do statistics for training
            train_loss += loss.data[0]

        # Save a temp model
        # torch.save(net, temp_model_name)
        net.save(epoch, train_loss / opt.NUM_TRAIN, "temp_model.dat")

        # Start testing
        test_loss = testing(opt, test_loader, net)

        writer.add_scalar("Train/loss", train_loss / opt.NUM_TRAIN, epoch + PRE_EPOCH)
        writer.add_scalar("Test/loss", test_loss / opt.NUM_TEST, epoch + PRE_EPOCH)
        # Output results
        print('Epoch [%d/%d], Train Loss: %.4f Test Loss: %.4f'
            % (PRE_EPOCH + epoch + 1, opt.NUM_EPOCHS, train_loss / opt.NUM_TRAIN,
               test_loss / opt.NUM_TEST))
        vec_dif(outputs, labels)

        # Save the best model
        if test_loss / opt.NUM_TEST < best_loss:
            best_loss = test_loss / opt.NUM_TEST
            # torch.save(net, best_model_name)
            net.save(epoch, train_loss / opt.NUM_TRAIN, "best_model.dat")

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
        loss = border_loss(outputs, labels, opt)
        test_loss += loss.data[0]

    return test_loss


def test_all(opt, all_loader, net, results):
    net.eval()
    test_loss = 0

    for i, data in tqdm(enumerate(all_loader), desc="Testing", total=len(all_loader), leave=False, unit='b'):
        inputs, labels, *_ = data
        if opt.USE_CUDA:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # Compute the outputs and judge correct
        outputs = net(inputs)
        loss = border_loss(outputs, labels, opt)
        test_loss += loss.data[0]
        if opt.USE_CUDA:
            outputs, labels = outputs.cpu().data.tolist(), labels.cpu().data.tolist()
        else:
            outputs, labels = outputs.data.tolist(), labels.data.tolist()
        results.extend([(label, output) for label, output in zip(labels, outputs)])

    print('==> Testing finished. You can find the result matrix in ./source/val_results/results.pkl')
    return results