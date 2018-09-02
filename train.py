# coding: utf-8
# Author: Zhongyang Zhang

import torch
import torch.nn as nn
import torch.autograd
import numpy as np
import threading
from torch.autograd import Variable
from tqdm import tqdm

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


def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


def vec_similarity(matrix_a, matrix_b):
    return (torch.sum(torch.pow(matrix_a - matrix_b, 2))).sqrt()


def vec_dif(matrix_a, matrix_b):
    print("Different between preds and labels is:", torch.mean(torch.abs(matrix_a - matrix_b)).data.tolist())


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


def training(opt, writer, train_loader, test_loader, net, pre_epoch, device, best_loss=100):
    best_loss = float(best_loss)
    threads = []

    # WARNING: input shape: (batch, 9, 41) but output shape: (batch, 41,9)
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.LEARNING_RATE, weight_decay=opt.WEIGHT_DECAY)

    for epoch in range(opt.NUM_EPOCHS):
        train_loss = 0

        # Start training
        net.train()

        print('==> Preparing Data ...')
        for i, data in tqdm(enumerate(train_loader), desc="Training", total=len(train_loader), leave=False, unit='b'):
            inputs, labels, *_ = data
            inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = border_loss(outputs, labels, opt)

            # loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Do statistics for training
            train_loss += loss.data[0]

        # Start testing
        test_loss = testing(opt, test_loader, net, device)

        writer.add_scalar("Train/loss", train_loss / opt.NUM_TRAIN, epoch + pre_epoch)
        writer.add_scalar("Test/loss", test_loss / opt.NUM_TEST, epoch + pre_epoch)
        # Output results
        print('Epoch [%d/%d], Train Loss: %.4f Test Loss: %.4f'
              % (pre_epoch + epoch + 1, opt.NUM_EPOCHS, train_loss / opt.NUM_TRAIN,
                 test_loss / opt.NUM_TEST))
        vec_dif(outputs, labels)

        if epoch > 0:
            threads[epoch - 1].join()
            best_loss_temp = threads[epoch - 1].best_loss
            if best_loss_temp != best_loss:
                print("==> Best Model Renewed. Best loss: {}".format(best_loss_temp))
            best_loss = best_loss_temp
        threads.append(MyThread(opt, net, epoch, train_loss, best_loss, test_loss))
        threads[epoch].start()
    print('==> Training Finished.')
    return net


def testing(opt, test_loader, net, device):
    net.eval()
    test_loss = 0

    for i, data in tqdm(enumerate(test_loader), desc="Testing", total=len(test_loader), leave=False, unit='b'):
        inputs, labels, *_ = data
        inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))

        # Compute the outputs and judge correct
        outputs = net(inputs)
        loss = border_loss(outputs, labels, opt)
        test_loss += loss.data[0]

    return test_loss


def test_all(opt, all_loader, net, results, device):
    net.eval()
    test_loss = 0

    for i, data in tqdm(enumerate(all_loader), desc="Testing", total=len(all_loader), leave=False, unit='b'):
        inputs, labels, *_ = data
        inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))

        # Compute the outputs and judge correct
        outputs = net(inputs)
        loss = border_loss(outputs, labels, opt)
        test_loss += loss.data[0]
        if opt.USE_CUDA:
            outputs, labels = outputs.cpu().data.tolist(), labels.cpu().data.tolist()
        else:
            outputs, labels = outputs.data.tolist(), labels.data.tolist()
        results.extend([(label, output) for label, output in zip(labels, outputs)])

    out_file = './source/val_results/' + opt.MODEL + '_' + opt.PROCESS_ID + '_results.pkl'
    print('==> Testing finished. You can find the result matrix in ' + out_file)
    return results
