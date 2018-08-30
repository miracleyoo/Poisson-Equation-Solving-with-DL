# coding: utf-8
# Author: Zhongyang Zhang

import os
import scipy.io
import pickle
import h5py
import warnings

warnings.filterwarnings("ignore")


def load_data(opt, root='./Datasets/'):
    """
    :Outputs:
        train_pairs : the path of the train  images and their labels' index list
        test_pairs  : the path of the test   images and their labels' index list
        class_names : the list of classes' names
    :param root : the root location of the dataset.

    Data Structure:
    train_data: dictionary, contains X_train and Y_train
    train_data['X_train'] :(6716, 2, 9, 41) num x channel x height x width
    train_data['Y_train'] :(6716, 369)

    """
    if opt.USE_NEW_DATA:
        DATA_PATH = [root + 'train_data_2.pkl', root + 'test_data_2.pkl']
        train_pairs = pickle.load(open(DATA_PATH[0], 'rb'))
        test_pairs = pickle.load(open(DATA_PATH[1], 'rb'))
        print("==> Load train data successfully.")
        print("==> Load test data successfully.")
        return train_pairs, test_pairs
    else:
        DATA_PATH = [root + 'train_data.mat', root + 'test_data.mat']

    try:
        train_data = scipy.io.loadmat(DATA_PATH[0])
    except NotImplementedError:
        train_data = h5py.File(DATA_PATH[0], 'r')
    finally:
        print("==> Load train data successfully.")

    try:
        test_data = scipy.io.loadmat(DATA_PATH[1])
    except NotImplementedError:
        test_data = h5py.File(DATA_PATH[1], 'r')
    finally:
        print("==> Load test data successfully.")

    train_data = dict((key,value) for key,value in train_data.items() if key=='X_train' or key=='Y_train')
    test_data = dict((key, value) for key, value in test_data.items() if key == 'X_test' or key == 'Y_test')
    train_pairs = [(x, y) for x, y in zip(train_data['X_train'], train_data['Y_train'])]
    test_pairs = [(x, y) for x, y in zip(test_data['X_test'], test_data['Y_test'])]

    return train_pairs, test_pairs


def load_all_data(root='./Datasets/'):
    """
    :Outputs:
        train_pairs : the path of the train  images and their labels' index list
        test_pairs  : the path of the test   images and their labels' index list
        class_names : the list of classes' names
    :param root : the root location of the dataset.
    """
    DATA_PATH = root+'all_data.mat'
    all_data = scipy.io.loadmat(DATA_PATH)

    all_data = dict((key,value) for key,value in all_data.items() if key=='X_all' or key=='Y_all')
    all_pairs = [(x, y) for x, y in zip(all_data['X_all'], all_data['Y_all'])]

    return all_pairs


def folder_init(opt):
    """
    Initialize folders required
    """
    if not os.path.exists('source'):
        os.mkdir('source')
    if not os.path.exists('source/reference'):
        os.mkdir('source/reference')
    if not os.path.exists('./source/summary/'):
        os.mkdir('./source/summary/')
    if not os.path.exists('./source/val_results/'):
        os.mkdir('./source/val_results/')
    if not os.path.exists('source/simulation_res'):
        os.mkdir('source/simulation_res')
    if not os.path.exists('source/simulation_res/intermediate_file'):
        os.mkdir('source/simulation_res/intermediate_file')
    if not os.path.exists('source/simulation_res/train_data'):
        os.mkdir('source/simulation_res/train_data')
    if not os.path.exists(opt.NET_SAVE_PATH):
        os.mkdir(opt.NET_SAVE_PATH)
