# coding: utf-8
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import DataLoader
from utils import *
from data_loader import *
from train import *
from config import Config
from models import miracle_net

opt = Config()

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((opt.PIC_SIZE, opt.PIC_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((opt.PIC_SIZE, opt.PIC_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

folder_init(opt)
# gen_name(opt)
train_pairs, test_pairs = load_data('./TempData/')

trainDataset = POISSON(train_pairs, opt)
train_loader = DataLoader(dataset=trainDataset, batch_size=opt.BATCH_SIZE, shuffle=True, num_workers=opt.NUM_WORKERS, drop_last=False)

testDataset  = POISSON(test_pairs, opt)
test_loader  = DataLoader(dataset=testDataset,  batch_size=opt.TEST_BATCH_SIZE, shuffle=False, num_workers=opt.NUM_WORKERS, drop_last=False)

opt.NUM_TRAIN    = len(trainDataset)
opt.NUM_TEST     = len(testDataset)

net = miracle_net.MiracleNet(opt)#models.resnet152(pretrained=False)
# net.fc = nn.Linear(8192, opt.NUM_CLASSES)
net = training(opt, train_loader, test_loader, net)

