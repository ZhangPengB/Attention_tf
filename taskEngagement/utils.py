import pandas as pd
from tensorflow.keras import utils as np_utils

import numpy as np
import scipy.io as scio
import os


def dim22dim3(dim2_data, epoch, fs, channels):
    sample_per = int(epoch * fs)  # 每个sample包含的采样点数
    time_len = dim2_data.shape[1] // sample_per
    dim3_data = np.zeros((time_len, channels, sample_per)).astype(np.float32)
    for i in range(time_len):
        dim3_data[i] = dim2_data[:, sample_per * i:sample_per + sample_per * i]

    return dim3_data


def produce_data(dirs, matfiles, fs=128, epoch=1, channels=64):
    '''
    读mat文件，截取各自10分钟数据，并按6：2：2产生数据集
    :param matfiles:
    :param fs:采样率default=128
    :param epoch:
    :return:
    '''
    train_data = np.array([[]] * 64)
    valid_data = np.array([[]] * 64)
    test_data = np.array([[]] * 64)
    # print(train_data.shape)
    for i in range(3):
        data = scio.loadmat(dirs + matfiles[i])
        keys = list(data.keys())
        tenMindData = data[keys[-1]][:, :10 * fs * 60]
        train_data = np.append(train_data, tenMindData[:, :int(10 * fs * 60 * 0.6)], axis=1)
        valid_data = np.append(valid_data, tenMindData[:, int(10 * fs * 60 * 0.6):int(10 * fs * 60 * 0.8)], axis=1)
        test_data = np.append(test_data, tenMindData[:, int(10 * fs * 60 * 0.8):], axis=1)

    print(train_data.shape)
    print(valid_data.shape)
    print(test_data.shape)

    train_X = dim22dim3(train_data, epoch, fs, channels)
    print(train_X.shape)
    train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1], train_X.shape[2])

    valid_X = dim22dim3(valid_data, epoch, fs, channels)
    valid_X = valid_X.reshape(valid_X.shape[0], 1, valid_X.shape[1], valid_X.shape[2])

    test_X = dim22dim3(test_data, epoch, fs, channels)
    test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1], test_X.shape[2])

    print(train_X.shape)
    print(valid_X.shape)
    print(test_X.shape)

    return train_X, valid_X, test_X


def produce_data_train_test(dirs, matfiles, fs=128, epoch=1, channels=64):
    '''
    读mat文件，截取各自10分钟数据，并按6：2：2产生数据集
    :param matfiles:
    :param fs:采样率default=128
    :param epoch:
    :return:
    '''
    train_data = np.array([[]] * 64)
    valid_data = np.array([[]] * 64)
    test_data = np.array([[]] * 64)
    # print(train_data.shape)
    for i in range(3):
        data = scio.loadmat(dirs + matfiles[i])
        keys = list(data.keys())
        tenMindData = data[keys[-1]][:, :10 * fs * 60]
        train_data = np.append(train_data, tenMindData[:, :int(10 * fs * 60 * 0.8)], axis=1)
        valid_data = np.append(valid_data, tenMindData[:, int(10 * fs * 60 * 0.8):int(10 * fs * 60 * 0.9)], axis=1)
        test_data = np.append(test_data, tenMindData[:, int(10 * fs * 60 * 0.9):], axis=1)

    # print(train_data.shape)
    # print(valid_data.shape)
    # print(test_data.shape)

    train_X = dim22dim3(train_data, epoch, fs, channels)
    # print(train_X.shape)
    train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1], train_X.shape[2])

    valid_X = dim22dim3(valid_data, epoch, fs, channels)
    valid_X = valid_X.reshape(valid_X.shape[0], 1, valid_X.shape[1], valid_X.shape[2])

    test_X = dim22dim3(test_data, epoch, fs, channels)
    test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1], test_X.shape[2])

    return train_X, valid_X, test_X


def produce_label_train_test():
    train_label = np.concatenate([np.ones((480, 1)) + 1, np.zeros((480, 1)), np.ones((480, 1))])
    valid_label = np.concatenate([np.ones((60, 1)) + 1, np.zeros((60, 1)), np.ones((60, 1))])
    test_label = np.concatenate([np.ones((60, 1)) + 1, np.zeros((60, 1)), np.ones((60, 1))])
    y_train = np_utils.to_categorical(train_label)  # Y_train.shape=(1080, 3)
    y_valid = np_utils.to_categorical(valid_label)
    y_test = np_utils.to_categorical(test_label)

    return y_train, y_valid, y_test


def produce_label():
    train_label = np.concatenate([np.ones((360, 1)) + 1, np.zeros((360, 1)), np.ones((360, 1))])
    valid_label = np.concatenate([np.ones((120, 1)) + 1, np.zeros((120, 1)), np.ones((120, 1))])
    test_label = np.concatenate([np.ones((120, 1)) + 1, np.zeros((120, 1)), np.ones((120, 1))])
    y_train = np_utils.to_categorical(train_label)  # Y_train.shape=(1080, 3)
    y_valid = np_utils.to_categorical(valid_label)
    y_test = np_utils.to_categorical(test_label)
    return y_train, y_valid, y_test


def write_predicition_message(file_path, message):
    f = open(file_path, 'a')
    f.write(message)
    f.write('\n')
    f.close()


# dirs = './datasets/wangshenhong/'
# matfiles = ['wangqifei.mat', 'wangjiangluo0.mat', 'wangjiangluo1.mat']
# train_data, valid_data, test_data = produce_data(dirs, matfiles, epoch=1)
