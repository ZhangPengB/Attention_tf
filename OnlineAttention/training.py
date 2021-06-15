import pandas as pd
from pyriemann.utils.viz import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os

from taskEngagement.utils import produce_label, produce_data, produce_data_train_test, produce_label_train_test
from tensorflow.keras.callbacks import ModelCheckpoint
from taskEngagement.EEGModels import EEGNet, ShallowConvNet, DeepConvNet
from taskEngagement.ACNN import DeepConvNet_V1

# epoch, fs, channels = 1, 128, 64
# sample_per = epoch * fs  # 每个样本包含的采样点数
# names = ['moderate', 'low', 'high']
# eegnet_flag = 3  # 加载模型标志

# 模型参数
# batch_size, epochs = 16, 110

# 加载数据
subject = 'zhaozhuren'
# dirs = './datasets/' + subject + '/'
# matfiles = ['zhaoqifei.mat', 'zhaojiangluo0.mat', 'zhaojiangluo1.mat']
train_data, valid_data, test_data = produce_data_train_test(dirs, matfiles)
# # train_data, valid_data, test_data = produce_data(dirs, matfiles)
# print(train_data.shape)
# print(valid_data.shape)
# print(test_data.shape)
#
# y_train, y_valid, y_test = produce_label_train_test()
# # y_train, y_valid, y_test = produce_label()
# print(y_train.shape)

model = DeepConvNet_V1(nb_classes=3, Chans=64, Samples=128,
                       dropoutRate=0.5)
# model_name = "AeentionNet"
filepath = '/tmp/DeepConvnetV1_checkPoints/' + subject + '/checkpoint.h5'

# compile the model and set the optimizers
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# count number of parameters in the model
numParams = model.count_params()

# set a valid path for your system to record model checkpoints,verbose=1 表示输出模型保存的信息
checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,
                               save_best_only=True)
print("----------dene-------")



# the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
# the weights all to be 1
class_weights = {0: 1, 1: 1, 2: 1}

################################################################################
# fit the model. Due to very small sample sizes this can get
# pretty noisy run-to-run, but most runs should be comparable to xDAWN +
# Riemannian geometry classification (below)
################################################################################
fittedModel = model.fit(train_data, y_train, batch_size=16, epochs=epochs,
                        verbose=1, validation_data=(valid_data, y_valid),
                        callbacks=[checkpointer], class_weight=class_weights)

##############################################################################
# can alternatively used the weights provided in the repo. If so it should get
# you 93% accuracy. Change the WEIGHTS_PATH variable to wherever it is on your
# system.
###############################################################################


# WEIGHTS_PATH = /path/to/EEGNet-8-2-weights.h5
# model.load_weights(WEIGHTS_PATH)

# file_path = "./results"
# f = open(file_path, 'a')
# f.write("##############" + model_name + "#############" + '\n' + "subject:" + subject + '\n'
#         + "epochs:" + str(epochs) + "\n" + "batch_size:" + str(batch_size) + "\n")
# f.close()
#
# loss=fittedModel.history['loss']
# ax1=plt.subplot()
# ax1.set_xlabel("iteration")
# ax1.set_ylabel("training_loss")
# plt.plot(loss,label='Training loss')
# plt.title("attentionNet training loss")
# plt.legend(loc=0,ncol=1)
# plt.show()