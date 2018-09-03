# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

from torch.utils.data import DataLoader
from utils import *
from data_loader import *
from train import *
from config import Config
from tensorboardX import SummaryWriter
from models import miracle_net, miracle_wide_net, miracle_weight_wide_net, miracle_lineconv_net
import pickle
import torch
import argparse
import warnings

warnings.filterwarnings("ignore")


def model_to_device(model, device):
    # Data Parallelism
    if torch.cuda.device_count() > 1:
        print("==> Using", torch.cuda.device_count(), "GPUs.")
        model = nn.DataParallel(model)
    model.to(device)
    return model


def load_model(opt, model, device, model_type):
    print('==> Now using ' + opt.MODEL + '_' + opt.PROCESS_ID)
    print('==> Loading model ...')

    net_save_prefix = opt.NET_SAVE_PATH + opt.MODEL + '_' + opt.PROCESS_ID + '/'
    temp_model_name = net_save_prefix + model_type
    if not os.path.exists(net_save_prefix):
        os.mkdir(net_save_prefix)
    if os.path.exists(temp_model_name):
        model, PRE_EPOCH, best_loss = model.load(temp_model_name)
        print("Load existing model: %s" % temp_model_name)
    else:
        raise FileNotFoundError("The model you want to load doesn't exist!")
    model_to_device(model, device)
    return model, PRE_EPOCH, best_loss


def main():
    # Initializing configs
    allDataset = None
    all_loader = None
    # opt = Config()
    folder_init(opt)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    train_pairs, test_pairs = load_data(opt, './TempData/')

    trainDataset = POISSON(train_pairs, opt)
    train_loader = DataLoader(dataset=trainDataset, batch_size=opt.BATCH_SIZE, shuffle=True,
                              num_workers=opt.NUM_WORKERS, drop_last=False)

    testDataset = POISSON(test_pairs, opt)
    test_loader = DataLoader(dataset=testDataset, batch_size=opt.TEST_BATCH_SIZE, shuffle=False,
                             num_workers=opt.NUM_WORKERS, drop_last=False)

    if opt.TRAIN_ALL or opt.TEST_ALL:
        train_pairs.extend(test_pairs)
        all_pairs = train_pairs
        allDataset = POISSON(all_pairs, opt)
        all_loader = DataLoader(dataset=allDataset, batch_size=opt.TEST_BATCH_SIZE, shuffle=True,
                                num_workers=opt.NUM_WORKERS, drop_last=False)

    opt.NUM_TEST = len(testDataset)
    print("==> All datasets are generated successfully.")

    # Initialize model chosen
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

    # Instantiation of tensorboard and add net graph to it
    writer = SummaryWriter(opt.SUMMARY_PATH)
    dummy_input = Variable(torch.rand(opt.BATCH_SIZE, 2, 9, 41))
    writer.add_graph(net, dummy_input)

    # Start training or testing
    if opt.TEST_ALL:
        results = []
        net, *_ = load_model(opt, net, device, "best_model.dat")
        results = test_all(opt, all_loader, net, results, device)
        out_file = './source/val_results/' + opt.MODEL + '_' + opt.PROCESS_ID + '_results.pkl'
        pickle.dump(results, open(out_file, 'wb+'))
    else:
        pre_epoch = 0
        best_loss = 100
        if opt.LOAD_SAVED_MOD:
            try:
                net, pre_epoch, best_loss = load_model(opt, net, device, "temp_model.dat")
            except FileNotFoundError:
                net = model_to_device(net, device)
        else:
            net = model_to_device(net, device)
        if opt.TRAIN_ALL:
            opt.NUM_TRAIN = len(allDataset)
            _ = training(opt, writer, all_loader, test_loader, net, pre_epoch, device, best_loss)
        else:
            opt.NUM_TRAIN = len(trainDataset)
            _ = training(opt, writer, train_loader, test_loader, net, pre_epoch, device, best_loss)


if __name__ == '__main__':
    # Options
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-lsm', '--LOAD_SAVED_MOD', type=bool,
                        help='If you want to load saved model')
    parser.add_argument('-und', '--USE_NEW_DATA', type=bool,
                        help='If you want to use new data')
    parser.add_argument('-tra', '--TRAIN_ALL',  type=bool,
                        help='If you want to train full data(including test data)')
    parser.add_argument('-tea', '--TEST_ALL', type=bool,
                        help='If you want to test full data')
    parser.add_argument('-wd', '--WEIGHT_DECAY', type=float,
                        help='Weight decay of L2 regularization')
    parser.add_argument('-lr', '--LEARNING_RATE', type=float,
                        help='Learning rate')
    parser.add_argument('-l', '--LENGTH', type=float,
                        help='Length of the matrix')
    parser.add_argument('-w', '--WIDTH', type=float,
                        help='Width of the matrix')
    args = parser.parse_args()
    print(args)
    opt = Config()
    for k, v in vars(args).items():
        if v is not None:
            setattr(opt, k, v)
    main()
