import numpy as np
import scipy.io as scio


# train_data = np.array([[]] * 64)
# print(train_data.shape)


def readMat(matfiles, fs=128):
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
    print(train_data.shape)
    for i in range(3):
        data = scio.loadmat(matfiles[i])
        keys = list(data.keys())
        tenMindData = data[keys[-1]][:, :10 * fs * 60]
        # print(tenMindData.shape)
        # print(tenMindData[:, :int(10 * fs * 60 * 0.6)])
        train_data = np.append(train_data, tenMindData[:, :int(10 * fs * 60 * 0.6)], axis=1)
        valid_data = np.append(valid_data, tenMindData[:, int(10 * fs * 60 * 0.6):int(10 * fs * 60 * 0.8)], axis=1)
        test_data = np.append(test_data, tenMindData[:,int(10 * fs * 60 * 0.8):], axis=1)

    print(train_data.shape)
    print(valid_data.shape)
    print(test_data.shape)
    return train_data,valid_data,test_data


print("---")
matfiles = ['../datasets/qifei.mat', '../datasets/jiangluo0.mat', '../datasets/jiangluo1.mat']
train_data,valid_data,test_data=readMat(matfiles)
