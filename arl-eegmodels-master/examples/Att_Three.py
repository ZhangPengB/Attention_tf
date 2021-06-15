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

def dim22dim3(dim2_data):
    time_len = dim2_data.shape[1] // 128
    dim3_data = np.zeros((time_len, 18, 128)).astype(np.float32)
    for i in range(time_len):
        dim3_data[i] = dim2_data[:, 128 * i:128 + 128 * i]
    return dim3_data


file = scipy.io.loadmat('./eeg_record34.mat')  # change filename accordingly
print(file.keys())  # file is a nested dictionary, data is in 'o' key
# data = pd.DataFrame.from_dict(file["o"]["data"][0, 0])
data = pd.DataFrame(file['o']['data'][0, 0])
data = data.loc[:, 3:20].values
print(data.shape)  # (357224, 18)

# 所有数据先高通滤波
# data = bandpass_cnt(data, 14, 49, 128, filt_order=3, axis=1)  # data.shape=(308868, 18)
# print("data.shape=", data.shape)v

# print("data.T.shape",data.T.shape)
# data=data.T
# data.reshape((2794,18,128))
# 转为三维
data = dim22dim3(data.T)  # data.shape=(2413, 18, 128)
print(data.shape)

focused_data = data[:10 * 60, :, :]
unfocused_data = data[10 * 60:20 * 60, :, :]
dowsing_data = data[20 * 60:30 * 60, :]
print(focused_data.shape)
print(unfocused_data.shape)
print(dowsing_data.shape)

pred_data = data[:30 * 60, :, :]  # pred_data.shape=(1200, 18, 128)
print("pred_data,", pred_data.shape)

# 数据标签,focused为1,unfocused为0，dowsing为2
focused_data_label = np.ones(focused_data.shape[0])
unfocused_data_label = np.zeros(unfocused_data.shape[0])
dowsing_data_label = np.ones(dowsing_data.shape[0]) + 1
pred_data_label = np.hstack(
    (focused_data_label, unfocused_data_label, dowsing_data_label))  # pred_data_label.shape=(1200,)
print(pred_data_label)
print(pred_data_label.shape)

# 打乱所有数据和标签
index = [i for i in range(len(pred_data))]
random.shuffle(index)  # 打乱行索引
# print(index)
pred_data = pred_data[index]
print(pred_data_label[100:200])
pred_data_label = pred_data_label[index]
print(pred_data_label[100:200])

# 训练数据:60%,测试数据：20%，验证数据：20%
pred_len = len(pred_data)
print("pred_len", pred_len)
train_data = pred_data[:int(pred_len * 0.6)]  # (1080, 18, 128)
valid_data = pred_data[int(pred_len * 0.6):int(pred_len * 0.8)]  # (360, 18, 128)
test_data = pred_data[int(pred_len * 0.8):]  # (360, 18, 128)
print("train", train_data.shape)
print("train", valid_data.shape)
print("train", test_data.shape)

train_label = pred_data_label[:int(pred_len * 0.6)]  # train_label.shape=(1080,)
valid_label = pred_data_label[int(pred_len * 0.6):int(pred_len * 0.8)]  # valid_label.shape=(360,)
test_label = pred_data_label[int(pred_len * 0.8):]  # test_label.shape=(360,)
print(train_label.shape, type(train_label))
print(valid_label.shape)
print(test_label.shape)

kernels, chans, samples = 1, 18, 128

X_train = train_data.reshape(train_data.shape[0], kernels, chans, samples)  # X_train shape: (1080, 1, 18, 128)
X_valid = valid_data.reshape(valid_data.shape[0], kernels, chans, samples)
X_test = test_data.reshape(test_data.shape[0], kernels, chans, samples)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

print(train_label[:10])
# 转换为独热编码
Y_train = np_utils.to_categorical(train_label)  # Y_train.shape=(1080, 3)
print(Y_train[:10], Y_train.shape)
Y_validate = np_utils.to_categorical(valid_label)
Y_test = np_utils.to_categorical(test_label)

# configure the EEGNet-8,2,16 model with kernel length of 32 samples (other
# model configurations may do better, but this is a good starting point)
model = EEGNet(nb_classes=3, Chans=chans, Samples=samples,
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
                        verbose=2, validation_data=(X_valid, Y_validate),
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

print("probs.shape=", probs.shape)
print(probs[:100])
preds = probs.argmax(axis=-1)
print(preds[:100])
acc = np.mean(preds == Y_train.argmax(axis=-1))
print("Classification accuracy train: %f " % (acc))

probs = model.predict(X_valid)
preds = probs.argmax(axis=-1)
acc = np.mean(preds == Y_validate.argmax(axis=-1))
print("Classification accuracy valid: %f " % (acc))

probs = model.predict(X_test)
preds = probs.argmax(axis=-1)
acc = np.mean(preds == Y_test.argmax(axis=-1))
print("Classification accuracy  test: %f " % (acc))

############################# PyRiemann Portion ##############################

# code is taken from PyRiemann's ERP sample script, which is decoding in
# the tangent space with a logistic regression

n_components = 2  # pick some components

# set up sklearn pipeline
clf = make_pipeline(XdawnCovariances(n_components),
                    TangentSpace(metric='riemann'),
                    LogisticRegression())

preds_rg = np.zeros(len(Y_test))

# reshape back to (trials, channels, samples)
X_train = X_train.reshape(X_train.shape[0], chans, samples)
X_test = X_test.reshape(X_test.shape[0], chans, samples)
X_valid = X_valid.reshape(X_valid.shape[0], chans, samples)
print("X_train.shape=", X_train.shape)
print("X_test.shape=", X_test.shape)
print("X_valid.shape=", X_valid.shape)

# train a classifier with xDAWN spatial filtering + Riemannian Geometry (RG)
# labels need to be back in single-column format
clf.fit(X_train, Y_train.argmax(axis=-1))

preds_rg = clf.predict(X_test)
# Printing the results
acc2 = np.mean(preds_rg == Y_test.argmax(axis=-1))
print("Classification accuracy RG_test: %f " % (acc2))

preds_rg = clf.predict(X_train)
acc2 = np.mean(preds_rg == Y_train.argmax(axis=-1))
print("Classification accuracy RG_train: %f " % (acc2))

preds_rg = clf.predict(X_valid)
acc2 = np.mean(preds_rg == Y_validate.argmax(axis=-1))
print("Classification accuracy RG_valid: %f " % (acc2))

# plot the confusion matrices for both classifiers
# names = ['audio left', 'audio right', 'vis left', 'vis right']
names = ['focused', 'unfocused', 'drowsing']

plt.figure(0)
plot_confusion_matrix(preds, Y_test.argmax(axis=-1), names, title='EEGNet-8,2')
plt.show()

plt.figure(1)
plot_confusion_matrix(preds_rg, Y_test.argmax(axis=-1), names, title='xDAWN + RG')
plt.show()
