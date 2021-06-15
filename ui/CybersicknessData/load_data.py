import os
import scipy.io as sco
import random
import numpy as np


def load_data(path, _class):
    '''

    :param path: 要加载的文件夹路径
    :param _class: 数据类别文件夹
    :return:list - put data to a list
    '''
    class_path = path + _class  # + "180" + ".mat"
    dirs = os.listdir(class_path)  # [:-1]
    # print("dirs=",dirs)
    # print(len(dirs))
    # print(dirs)
    # data = np.zeros((64, 1))
    random.shuffle(dirs)  ## 随机排列文件
    # train_num = len(dir)*0.8-1
    # train_data = np.zeros((64, 1))
    # test_data = np.zeros((64, 1))
    # for i in range(len(dir)):
    #     path = class_path + '/' + dir[i]
    #     if i <= train_num:
    #         tmp_data = sco.loadmat(path)
    #         train_data = np.hstack((train_data,tmp_data[dir[i].split('.')[0]]))
    #         # train_data = np.hstack((train_data, tmp_data[_class]))
    #     else:
    #         tmp_data = sco.loadmat(path)
    #         test_data = np.hstack((test_data, tmp_data[dir[i].split('.')[0]]))
    data = []
    for i in range(len(dirs)):
        tmp_data = np.zeros((64, 1))
        tmp = sco.loadmat(class_path + '/' + dirs[i])
        tmp_data = np.hstack((tmp_data, tmp[_class]))
        data.append(tmp_data[:, 1:])

    # tmp = sco.loadmat(class_path + '/' + dirs[0])
    # data = np.hstack((data, tmp[_class]))

    # data = dim22dim3(data[:, 1:])

    # return data[:, 1:]

    return data

#
# print("----------------测试load_data()--------------")
# path = "E:/PyCharmProject/venv/egg/CybersicknessData/t2/"
# cyber_data = load_data(path, "cyber")
# silent_data = load_data(path, "silent")
# print("cyber_data.shape", cyber_data)
# print("silent_data.shape=", silent_data)


def train_data(data):
    '''
    将数据的60%作为训练数据
    :param data: 所有数据:[array([[....]])]
    :return: list - [array([[...]])]
    '''
    length = data[0].shape[1] // 1024
    t1 = int(length * 0.6)
    train_data = data[0][:, : t1 * 1024]
    if len(data) > 1:
        for i in range(1, len(data)):
            length = data[i].shape[1] // 1024
            t1 = int(length * 0.6)
            train_data = np.hstack((train_data, data[i][:, : t1 * 1024]))
    return train_data


# print("--------------测试train_data()-----------")


# train_data()

def valid_data(data):
    '''
    将数据的20%作为验证集
    :param data: list - 所有数据
    :return: list - 验证数据：[array([[...]])]
    '''
    length = data[0].shape[1] // 1024
    t1 = int(length * 0.6)
    t2 = int(length * 0.8)
    valid_data = data[0][:, t1 * 1024: t2 * 1024]
    if len(data) > 1:
        for i in range(1, len(data)):
            length = data[i].shape[1] // 1024
            t1 = int(length * 0.6)
            t2 = int(length * 0.8)
            valid_data = np.hstack((valid_data, data[i][:, t1 * 1024: t2 * 1024]))
    return valid_data


def test_data(data):
    '''
    将数据最后部分的20%作为测试集
    :param data: list - 所有数据：[array([[....]])]
    :return: list - 验证集：[array([[...
    ]])]
    '''
    length = data[0].shape[1] // 1024
    t2 = int(length * 0.8)
    test_data = data[0][:, t2 * 1024:]
    if len(data) > 1:
        for i in range(1, len(data)):
            length = data[i].shape[1] // 1024
            t2 = int(length * 0.8)
            test_data = np.hstack((test_data, data[i][:, t2 * 1024:]))
    return test_data


def dim22dim3(dim2_data):
    time_len = dim2_data.shape[1] // 1024
    dim3_data = np.zeros((time_len, 64, 1024)).astype(np.float32)
    for i in range(time_len):
        dim3_data[i] = dim2_data[:, 1024 * i:1024 + 1024 * i]
    return dim3_data


# if __name__ == "__main__":
#     path = "/home/frost/Cybersickness/CybersicknessData/t3/"
#     data = load_data(path, "cyber")
#     train_data = train_data(data)
#     valid_data = valid_data(data)
#     test_data = test_data(data)
#     # dim3_data = load_data(path, "cyber")
