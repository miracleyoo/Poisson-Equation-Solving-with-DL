# coding: utf-8
# Author: Zhongyang Zhang

import torch
from torch.utils.data import Dataset


class POISSON(Dataset):
    def __init__(self, data, opt):
        super(POISSON, self).__init__()
        self.data = data
        self.opt = opt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input, label = self.data[index]
        return torch.from_numpy(input).float(), torch.from_numpy(label).float()#.astype(np.float32)
