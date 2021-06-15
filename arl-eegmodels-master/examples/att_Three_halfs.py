import pandas as pd
from scipy import signal
import scipy.io
import numpy as np
import random
from numpy.random import RandomState
import tensorflow as tf
from tensorflow.keras import utils as np_utils
from EEGModels import EEGNet

from tensorflow.keras.callbacks import ModelCheckpoint

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt

# 自己加的
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


# {number-of-samples}x25  thus o.data(:,i) comprises one data channel
# 1-'EDCOUNTER' 2-'EDINTERPOLATED'
# 3-'EDRAWCQ'
# 4-'EDAF3' 5-'EDF7'
# 6-'EDF3' 7-'EDFC5'
# 8-'EDT7' 9-'EDP7'
# 10-'EDO1' 11-'EDO2'
# 12-'EDP8' 13-'EDT8'
# 14-'EDFC6' 15-'EDF4'
# 16-'EDF8' 17-'EDAF4'
# 18-'EDGYROX' 19-'EDGYROY'
# 20-'EDTIMESTAMP' 21-'EDESTIMESTAMP' 22-'EDFUNCID' 23-'EDFUNCVALUE' 24-'EDMARKER'
# 25'EDSYNCSIGNAL'
# The EEG data is in the channels 4:17.

# 5-'EDF7', 6-'EDF3' ,  8-'EDT7' 9-'EDP7', 10-'EDO1' 11-'EDO2',12-'EDP8' 13-'EDT8',17-'EDAF4'

def load_raw_data(path):
    '''
    读入指定路径下数据并转为ndarry
    :param path:
    :return:
    '''
    file = scipy.io.loadmat(path)  # change filename accordingly
    print(file.keys())  # file is a nested dictionary, data is in 'o' key
    # data = pd.DataFrame.from_dict(file["o"]["data"][0, 0])
    data = pd.DataFrame(file['o']['data'][0, 0])
    data = data.loc[:, 3:20].values
    return data


def dim22dim3(dim2_data, sample_intervel, fs, channels):
    '''
    将样本数据转为3维（samples,channels,fs)
    samples=采样总点/点数
    0.5秒为一个样本:sample=dim2_data/f2/2
    1秒为一个样本:sample=总样本点/fs
    :param dim2_data:
    :param sample_length:
    :return:
    '''
    sample_per = int(sample_intervel * fs)  # 每个sample包含的采样点数
    time_len = dim2_data.shape[1] // sample_per
    dim3_data = np.zeros((time_len, channels, sample_per)).astype(np.float32)
    for i in range(time_len):
        dim3_data[i] = dim2_data[:, sample_per * i:sample_per + sample_per * i]
    dim3_data = dim3_data[:int(30 * 60 // sample_intervel), :, :]
    return dim3_data


def produce_three_data_label(data, sample_intervel, kernels=1, sample_per=128, chans=25):
    '''
    产生train_data,valid_data,test_date和对应的lable
    :param data:
    :param sample_intervel: 样本时间间隔
    :param kernels: 卷积核
    :param sample_per: 每秒钟采集的点数
    :param chans: 通道数
    :return:
    '''
    focused_data = data[:int(10 * 60 // sample_intervel), :, :]
    unfocused_data = data[int(10 * 60 // sample_intervel):int(20 * 60 // sample_intervel), :, :]
    drowsing_data = data[int(20 * 60 // sample_intervel):int(30 * 60 // sample_intervel), :, :]

    focused_data_label = np.ones(focused_data.shape[0])
    unfocused_data_label = np.zeros(unfocused_data.shape[0])
    drowsing_data_label = np.ones(drowsing_data.shape[0]) + 1

    flen = len(focused_data)
    uflen = len(unfocused_data)
    dlen = len(drowsing_data)

    train_data = np.vstack((focused_data[:int(flen * 0.6)], unfocused_data[:int(uflen * 0.6)],
                            drowsing_data[:int(dlen * 0.6)]))
    valid_data = np.vstack((focused_data[int(flen * 0.6):int(flen * 0.8)],
                            unfocused_data[int(uflen * 0.6):int(uflen * 0.8)],
                            drowsing_data[int(dlen * 0.6):int(dlen * 0.8)]))
    test_data = np.vstack((focused_data[int(flen * 0.8):], unfocused_data[int(uflen * 0.8):],
                           drowsing_data[int(dlen * 0.8):]))

    X_train = train_data.reshape((train_data.shape[0], kernels, chans, sample_per))  # X_train shape: (1080, 1, 18, 128)
    X_valid = valid_data.reshape((valid_data.shape[0], kernels, chans, sample_per))
    X_test = test_data.reshape((test_data.shape[0], kernels, chans, sample_per))

    data_label = np.hstack(
        (focused_data_label, unfocused_data_label, drowsing_data_label))
    print("data_label.shape=", data_label.shape)
    label_len = len(data_label)

    train_label = np.hstack((focused_data_label[:int(flen * 0.6)], unfocused_data_label[:int(uflen * 0.6)],
                             drowsing_data_label[:int(dlen * 0.6)]))
    valid_label = np.hstack((focused_data_label[int(flen * 0.6):int(flen * 0.8)],
                             unfocused_data_label[int(uflen * 0.6):int(uflen * 0.8)],
                             drowsing_data_label[int(dlen * 0.6):int(dlen * 0.8)]))
    test_label = np.hstack((focused_data_label[int(flen * 0.8):], unfocused_data_label[int(uflen * 0.8):],
                            drowsing_data_label[int(dlen * 0.8):]))

    # print("train_label.shape", train_label.shape)
    # print("valid_label.shape", valid_label.shape)
    print(test_label.shape)
    # 转换为独热编码
    Y_train = np_utils.to_categorical(train_label)  # Y_train.shape=(1080, 3)
    # print(Y_train[:10], Y_train.shape)
    Y_valid = np_utils.to_categorical(valid_label)
    Y_test = np_utils.to_categorical(test_label)

    # print("Y_train.shape", Y_train.shape)

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test


def datasets(data, kernels=1, chans=18, sample_per=128):
    '''
    产生训练集，测试集和验证集
    训练数据:60%,测试数据：20%，验证数据：20%
    :param data:
    :return:
    '''
    data_len = len(data)
    # print("pred_len", data_len)
    train_data = data[:int(data_len * 0.6)]  # (1080, 18, 128)
    valid_data = data[int(data_len * 0.6):int(data_len * 0.8)]  # (360, 18, 128)
    test_data = data[int(data_len * 0.8):]  # (360, 18, 128)

    X_train = train_data.reshape(train_data.shape[0], kernels, chans, sample_per)  # X_train shape: (1080, 1, 18, 128)
    X_valid = valid_data.reshape(valid_data.shape[0], kernels, chans, sample_per)
    X_test = test_data.reshape(test_data.shape[0], kernels, chans, sample_per)
    return X_train, X_valid, X_test


# def dataset_label(label):
#     '''
#
#     :param label:
#     :return:
#     '''
#     # 训练数据:60%,测试数据：20%，验证数据：20%
#     label_len = len(label)
#     train_label = label[:int(label_len * 0.6)]  # train_label.shape=(1080,)
#     valid_label = label[int(label_len * 0.6):int(label_len * 0.8)]  # valid_label.shape=(360,)
#     test_label = label[int(label_len * 0.8):]  # test_label.shape=(360,)
#
#     # 转换为独热编码
#     Y_train = np_utils.to_categorical(train_label)  # Y_train.shape=(1080, 3)
#     # print(Y_train[:10], Y_train.shape)
#     Y_valid = np_utils.to_categorical(valid_label)
#     Y_test = np_utils.to_categorical(test_label)
#
#     print("Y_train.shape",Y_train.shape)
#     return Y_train, Y_valid, Y_test


# def shuffle_data_label(data, label):
#     index = [i for i in range(len(data))]
#     random.shuffle(index)  # 打乱行索引
#     data = data[index]
#     data_label = label[index]
#
#     return data, data_label


sample_intervel, fs, channels = 1, 128, 18
sample_per = int(sample_intervel * fs)  # 每个sample包含的采样点数
data = load_raw_data('./eeg_record34.mat')
print("raw_data.shape", data.shape)

dim3_data = dim22dim3(data.T, sample_intervel, fs, channels)
print(dim3_data.shape)

X_train, X_valid, X_test, Y_train, Y_valid, Y_test = produce_three_data_label(dim3_data, sample_intervel,
                                                                              kernels=1, sample_per=sample_per,
                                                                              chans=channels)

print(X_train.shape)  # (1080, 1, 25, 128)
print(X_valid.shape)  # (360, 1, 25, 128)
print(X_test.shape)  # (360, 1, 25, 128)
print(Y_train.shape)  # (1080, 3)
print(Y_valid.shape)  # (360, 3)
print(Y_test.shape)  # (360, 3)

model = EEGNet(nb_classes=3, Chans=channels, Samples=sample_per,
               dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,
               dropoutType='Dropout')

# compile the model and set the optimizers
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# count number of parameters in the model
numParams = model.count_params()

# set a valid path for your system to record model checkpoints
checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                               save_best_only=True)
print("----------dene-------")

###############################################################################
# if the classification task was imbalanced (significantly more trials in one
# class versus the others) you can assign a weight to each class during
# optimization to balance it out. This data is approximately balanced so we
# don't need to do this, but is shown here for illustration/completeness.
###############################################################################

# the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
# the weights all to be 1
class_weights = {0: 1, 1: 1, 2: 1}

################################################################################
# fit the model. Due to very small sample sizes this can get
# pretty noisy run-to-run, but most runs should be comparable to xDAWN +
# Riemannian geometry classification (below)
################################################################################
fittedModel = model.fit(X_train, Y_train, batch_size=16, epochs=50,
                        verbose=2, validation_data=(X_valid, Y_valid),
                        callbacks=[checkpointer], class_weight=class_weights)

# load optimal weights
model.load_weights('/tmp/checkpoint.h5')

###############################################################################
# can alternatively used the weights provided in the repo. If so it should get
# you 93% accuracy. Change the WEIGHTS_PATH variable to wherever it is on your
# system.
###############################################################################

# WEIGHTS_PATH = /path/to/EEGNet-8-2-weights.h5
# model.load_weights(WEIGHTS_PATH)

###############################################################################
# make prediction on test set.
###############################################################################

probs = model.predict(X_train)
preds = probs.argmax(axis=-1)
print("train preds=", preds[:100])
acc = np.mean(preds == Y_train.argmax(axis=-1))
print("Y_train_label:", Y_train.argmax(axis=-1)[:100])
print("Classification accuracy train: %f " % (acc))

probs = model.predict(X_valid)
preds = probs.argmax(axis=-1)
acc = np.mean(preds == Y_valid.argmax(axis=-1))
print("Classification accuracy valid: %f " % (acc))

probs = model.predict(X_test)
preds = probs.argmax(axis=-1)
# print("test preds=", preds[:100])
acc = np.mean(preds == Y_test.argmax(axis=-1))
print("Classification accuracy  test: %f " % (acc))

names = ['focused', 'unfocused', 'drowsing']
plt.figure(0)
plot_confusion_matrix(preds, Y_test.argmax(axis=-1), names, title='EEGNet-8,2')
plt.show()

############################# PyRiemann Portion ##############################

# code is taken from PyRiemann's ERP sample script, which is decoding in
# the tangent space with a logistic regression

# n_components = 2  # pick some components
#
# # set up sklearn pipeline
# clf = make_pipeline(XdawnCovariances(n_components),
#                     TangentSpace(metric='riemann'),
#                     LogisticRegression())
#
# preds_rg = np.zeros(len(Y_test))
#
# # reshape back to (trials, channels, samples)
# X_train = X_train.reshape(X_train.shape[0], channels, sample_per)
# X_test = X_test.reshape(X_test.shape[0], channels, sample_per)
# X_valid = X_valid.reshape(X_valid.shape[0], channels, sample_per)
# print("X_train.shape=", X_train.shape)
# print("X_test.shape=", X_test.shape)
# print("X_valid.shape=", X_valid.shape)
#
# # train a classifier with xDAWN spatial filtering + Riemannian Geometry (RG)
# # labels need to be back in single-column format
# clf.fit(X_train, Y_train.argmax(axis=-1))
#
# preds_rg = clf.predict(X_test)
# # Printing the results
# acc2 = np.mean(preds_rg == Y_test.argmax(axis=-1))
# print("Classification accuracy RG_test: %f " % (acc2))
#
# preds_rg = clf.predict(X_train)
# acc2 = np.mean(preds_rg == Y_train.argmax(axis=-1))
# print("Classification accuracy RG_train: %f " % (acc2))
#
# preds_rg = clf.predict(X_valid)
# acc2 = np.mean(preds_rg == Y_valid.argmax(axis=-1))
# print("Classification accuracy RG_valid: %f " % (acc2))
#
# # plot the confusion matrices for both classifiers
# # names = ['audio left', 'audio right', 'vis left', 'vis right']
# names = ['focused', 'unfocused', 'drowsing']
#
# plt.figure(0)
# plot_confusion_matrix(preds, Y_test.argmax(axis=-1), names, title='EEGNet-8,2')
# plt.show()
#
# plt.figure(1)
# plot_confusion_matrix(preds_rg, Y_test.argmax(axis=-1), names, title='xDAWN + RG')
# plt.show()
