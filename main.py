# coding: utf-8
# Author: Zhongyang Zhang

from torch.utils.data import DataLoader
from utils import *
from data_loader import *
from train import *
from config import Config
from models import miracle_net, miracle_wide_net, miracle_weight_wide_net, miracle_lineconv_net
import pickle

opt = Config()

folder_init(opt)
train_pairs, test_pairs = load_data('./TempData/')
all_pairs = load_all_data('./TempData/')

trainDataset = POISSON(train_pairs, opt)
train_loader = DataLoader(dataset=trainDataset, batch_size=opt.BATCH_SIZE, shuffle=True, num_workers=opt.NUM_WORKERS,
                          drop_last=False)

testDataset = POISSON(test_pairs, opt)
test_loader = DataLoader(dataset=testDataset, batch_size=opt.TEST_BATCH_SIZE, shuffle=False,
                         num_workers=opt.NUM_WORKERS, drop_last=False)

allDataset = POISSON(all_pairs, opt)
all_loader = DataLoader(dataset=allDataset, batch_size=opt.TEST_BATCH_SIZE, shuffle=False,
                        num_workers=opt.NUM_WORKERS, drop_last=False)

opt.NUM_TRAIN = len(trainDataset)
opt.NUM_TEST = len(testDataset)

if opt.MODEL == 'MiracleWeightWideNet':
    net = miracle_weight_wide_net.MiracleWeightWideNet(opt)
elif opt.MODEL == 'MiracleWideNet':
    net = miracle_wide_net.MiracleWideNet(opt)
elif opt.MODEL == 'MiracleNet':
    net = miracle_net.MiracleNet(opt)
elif opt.MODEL == 'MiracleLineConvNet':
    net = miracle_lineconv_net.MiracleLineConvNet(opt)

if opt.TEST_ALL:
    results = []
    NET_SAVE_PREFIX = "./source/trained_net/" + net.model_name
    temp_model_name = NET_SAVE_PREFIX + "/best_model.dat"
    if os.path.exists(temp_model_name):
        net, PRE_EPOCH, best_loss = net.load(temp_model_name)
        print("Load existing model: %s" % temp_model_name)
    results = test_all(opt, all_loader, net, results)
    out_file = './source/val_results/' + opt.MODEL + '_' + opt.PROCESS_ID + '_results.pkl'
    pickle.dump(results, open(out_file, 'wb+'))
else:
    if opt.TRAIN_ALL:
        net = training(opt, all_loader, test_loader, net)
    else:
        net = training(opt, train_loader, test_loader, net)
