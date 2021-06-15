import numpy as np
import scipy.io as scio
import os
from sklearn import preprocessing

from scipy.cluster.hierarchy import linkage


def dim22dim3(dim2_data, epoch, fs, channels):
    sample_per = int(epoch * fs)  # 每个sample包含的采样点数
    time_len = int(dim2_data.shape[1] // sample_per)
    dim3_data = np.zeros((time_len, channels, sample_per)).astype(np.float32)
    for i in range(time_len):
        dim3_data[i] = dim2_data[:, sample_per * i:sample_per + sample_per * i]

    return dim3_data


def produce_data_train_test(dirs, matfiles, fs=128, epoch=1, channels=64, timelen=10, divide=0.8):
    '''
    读mat文件，截取各自timelen分钟数据，并按divide比例划分数据形成train,valid,test
    :param dirs:
    :param matfiles:
    :param fs:
    :param epoch:
    :param channels:
    :param timelen:
    :param divide: 数据集划分比例，为0.6表示按6:2:2划分,为0.8表示按8:1:1划分
    :return:
    '''
    if divide == 0.8:
        divide2 = 0.9
    elif divide == 0.6:
        divide2 = 0.8

    train_data = np.array([[]] * 64)
    valid_data = np.array([[]] * 64)
    test_data = np.array([[]] * 64)
    # print(train_data.shape)
    for i in range(2):
        data = scio.loadmat(dirs + matfiles[i])
        keys = list(data.keys())
        timelenData = data[keys[-1]][:, :timelen * fs * 60]
        train_data = np.append(train_data, timelenData[:, :int(timelen * fs * 60 * divide)], axis=1)
        valid_data = np.append(valid_data,
                               timelenData[:, int(timelen * fs * 60 * divide):int(timelen * fs * 60 * divide2)], axis=1)
        test_data = np.append(test_data, timelenData[:, int(timelen * fs * 60 * divide2):], axis=1)

    print(train_data.shape)
    print(valid_data.shape)
    print(test_data.shape)

    train_X = dim22dim3(train_data, epoch, fs, channels)
    # print(train_X.shape)
    train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1], train_X.shape[2])

    valid_X = dim22dim3(valid_data, epoch, fs, channels)
    valid_X = valid_X.reshape(valid_X.shape[0], 1, valid_X.shape[1], valid_X.shape[2])

    test_X = dim22dim3(test_data, epoch, fs, channels)
    test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1], test_X.shape[2])

    return train_X, valid_X, test_X


def produce_label(timelen=30, divide=0.6):
    if divide == 0.6:
        divide2 = 0.2
    else:
        divide2 = 0.1
    train_len_per_class = (int)(timelen * divide * 60)
    valid_len_per_class = (int)(timelen * divide2 * 60)
    test_len_per_class = (int)(timelen * divide2 * 60)
    # print(train_len_per_class)
    # print(valid_len_per_class)

    train_label = np.concatenate([np.zeros((train_len_per_class, 1)), np.ones((train_len_per_class, 1))])
    valid_label = np.concatenate([np.zeros((valid_len_per_class, 1)), np.ones((valid_len_per_class, 1))])
    test_label = np.concatenate([np.zeros((test_len_per_class, 1)), np.ones((test_len_per_class, 1))])
    enc = preprocessing.OneHotEncoder(sparse=False)
    y_train = enc.fit_transform(train_label)
    y_valid = enc.fit_transform(valid_label)
    y_test = enc.fit_transform(test_label)
    return y_train, y_valid, y_test


# print("--------测试---------")
# # 加载数据
# subject = 'zhaozhonghua'
# dirs = './datasets/' + subject + '/'
# matfiles = ['zhao_P_1.mat', 'zhao_N_1.mat']
#
# epoch, fs, channels = 1, 128, 64
# sample_per = epoch * fs  # 每个样本包含的采样点数
#
# train_data, valid_data, test_data = produce_data_train_test(dirs, matfiles, timelen=20, divide=0.6)
# print(train_data.shape)
# print(valid_data.shape)
# print(test_data.shape)
#
# y_train, y_valid, y_test = produce_label(timelen=20, divide=0.6)
# print(y_train.shape[0])
# print(y_valid.shape)
# print(y_test.shape)
