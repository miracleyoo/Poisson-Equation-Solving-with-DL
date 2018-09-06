# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import torch
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"


class Config(object):
    def __init__(self):
        self.USE_CUDA            = torch.cuda.is_available()
        self.LOAD_SAVED_MOD      = True
        self.TEST_ALL            = False
        self.TRAIN_ALL           = False
        self.SAVE_TEMP_MODEL     = True
        self.USE_NEW_DATA        = True
        self.NUM_CHANNEL         = 2
        self.NET_SAVE_PATH       = "./source/trained_net/"
        self.MODEL               = 'MiracleWeightWideNet'
        self.PROCESS_ID          = 'PADDING_LOSS1-2_WEI4-2-1-1-NEW_GEN'
        if self.TRAIN_ALL:
            self.PROCESS_ID += '_TRAIN_ALL'
        self.SUMMARY_PATH        = "./source/summary/"+self.MODEL+'_'+self.PROCESS_ID+'_' +\
                                   str(self.NUM_CHANNEL)+'CHANNEL/'

        self.TRAIN_DATA_RATIO    = 0.7
        self.NUM_TEST            = 0
        self.NUM_TRAIN           = 0
        self.TOP_NUM             = 1

        self.NUM_EPOCHS          = 100
        self.NUM_CLASSES         = 369
        self.LEARNING_RATE       = 1e-3
        self.WEIGHT_DECAY        = 1e-4
        self.LINER_HID_SIZE      = 1024
        self.LENGTH              = 41
        self.WIDTH               = 9

        self.BATCH_SIZE          = 512
        self.TEST_BATCH_SIZE     = 512
        self.NUM_WORKERS         = 0
