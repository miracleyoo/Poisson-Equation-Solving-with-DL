# coding: utf-8
# Author: Zhongyang Zhang

from torch.utils.data import DataLoader
from utils import *
from data_loader import *
from train import *
from config import Config
from models import miracle_net, miracle_wide_net, miracle_weight_wide_net, miracle_lineconv_net
import pickle
import torch
import warnings
warnings.filterwarnings("ignore")

opt = Config()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def model_to_device(model):
    # Data Parallelism
    if torch.cuda.device_count() > 1:
        print("==> Using", torch.cuda.device_count(), "GPUs.")
        model = nn.DataParallel(model)

    model.to(device)
    return model


def load_model(model, model_type):
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
    model_to_device(model)
    return model, PRE_EPOCH, best_loss


folder_init(opt)
train_pairs, test_pairs = load_data(opt, './TempData/')
all_pairs = load_all_data('./TempData/')

trainDataset = POISSON(train_pairs, opt)
train_loader = DataLoader(dataset=trainDataset, batch_size=opt.BATCH_SIZE, shuffle=True, num_workers=opt.NUM_WORKERS,
                          drop_last=False)

testDataset = POISSON(test_pairs, opt)
test_loader = DataLoader(dataset=testDataset, batch_size=opt.TEST_BATCH_SIZE, shuffle=False,
                         num_workers=opt.NUM_WORKERS, drop_last=False)

allDataset = POISSON(all_pairs, opt)
all_loader = DataLoader(dataset=allDataset, batch_size=opt.TEST_BATCH_SIZE, shuffle=True,
                        num_workers=opt.NUM_WORKERS, drop_last=False)

print("==> All datasets are generated successfully.")

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
finally:
    print("==> Model initialized successfully.")


opt.NUM_TEST = len(testDataset)

if opt.TEST_ALL:
    results = []
    net, *_ = load_model(net, "best_model.dat")
    results = test_all(opt, all_loader, net, results, device)
    out_file = './source/val_results/' + opt.MODEL + '_' + opt.PROCESS_ID + '_results.pkl'
    pickle.dump(results, open(out_file, 'wb+'))
else:
    pre_epoch = 0
    best_loss = 100
    if opt.LOAD_SAVED_MOD:
        try:
            net, pre_epoch, best_loss = load_model(net, "temp_model.dat")
        except FileNotFoundError:
            model_to_device(net)
    else:
        model_to_device(net)
    if opt.TRAIN_ALL:
        opt.NUM_TRAIN = len(allDataset)
        net = training(opt, all_loader, test_loader, net, pre_epoch, device, best_loss)
    else:
        opt.NUM_TRAIN = len(trainDataset)
        net = training(opt, train_loader, test_loader, net, pre_epoch, device, best_loss)
