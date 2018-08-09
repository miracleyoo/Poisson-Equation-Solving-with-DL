# coding: utf-8
# Author: Zhongyang Zhang

import torch
import os
import torch.nn as nn
import torchvision.models as models

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Config(object):
    def __init__(self):
        self.USE_CUDA            = torch.cuda.is_available()
        self.NET_SAVE_PATH       = "./source/trained_net/"
        self.MODEL               = 'MiracleWeightWideNet'
        self.PROCESS_ID          = 'PADDING_LOSS1-2_WEI4-2-1-1'
        self.SUMMARY_PATH        = "./source/summary/"+self.MODEL+'_'+self.PROCESS_ID+'/'
        self.TEST_ALL            = False
        self.TRAINDATARATIO      = 0.7
        self.LOAD_SAVED_MOD      = True
        self.PIC_SIZE            = 256
        self.NUM_TEST            = 0
        self.NUM_TRAIN           = 0
        self.TOP_NUM             = 1
        self.NUM_EPOCHS          = 100
        # self.BATCH_SIZE          = 2
        # self.TEST_BATCH_SIZE     = 1
        # self.NUM_WORKERS         = 1
        self.NUM_CLASSES         = 369
        self.LEARNING_RATE       = 0.001
        self.LINER_HID_SIZE      = 1024
        self.LENGTH              = 41
        self.WIDTH               = 9

        self.BATCH_SIZE          = 32
        self.TEST_BATCH_SIZE     = 128
        self.NUM_WORKERS         = 4