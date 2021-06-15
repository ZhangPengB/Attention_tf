import pandas as pd
from pyriemann.utils.viz import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os

from tensorflow.keras.callbacks import ModelCheckpoint
from KDTest.PreHandleData import produce_label, produce_data_train_test
from KDTest.EEGModels import EEGNet, ShallowConvNet, DeepConvNet
from KDTest.ACNN import DeepConvNet_V1

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

divide_class=2
sample_len, fs, channels = 1, 128, 64
divide=0.8   #数据划分比例
timelen=10
model_flag = 1 # 加载模型标志

# 模型参数
batch_size, epochs = 16, 8

# 加载数据
subject = 'wuailin'
dirs = './datasets/' + subject + '/'
matfiles = ['wu_P_1.mat', 'wu_N_1.mat']

train_data, valid_data, test_data = produce_data_train_test(dirs, matfiles, timelen=timelen, divide=divide)
print(train_data.shape)
print(valid_data.shape)
print(test_data.shape)

y_train, y_valid, y_test = produce_label(timelen=timelen,divide=divide)
print(y_train.shape)
print(y_valid.shape)
print(y_test.shape)

assert (train_data.shape[0]==y_train.shape[0])
assert (valid_data.shape[0]==y_valid.shape[0])
assert (test_data.shape[0]==y_test.shape[0])

if model_flag == 1:
    model = EEGNet(nb_classes=divide_class, Chans=channels, Samples=fs,
                   dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16,
                   dropoutType='Dropout')
    model_name = "eegnet"
    filepath = '/tmp/eegnet_checkPoints/' + subject + '/checkpoint.h5'
elif model_flag == 0:
    model = ShallowConvNet(nb_classes=divide_class, Chans=channels, Samples=fs,
                           dropoutRate=0.5)
    model_name = "shallowConvNet"
    filepath = '/tmp/shallowConvnet_checkPoints/' + subject + '/checkpoint.h5'
elif model_flag == 2:
    model = DeepConvNet(nb_classes=divide_class, Chans=channels, Samples=fs,
                        dropoutRate=0.5)
    model_name = "DeepConvNet"
    filepath = '/tmp/DeepConvnet_checkPoints/' + subject + '/checkpoint.h5'
else:
    model = DeepConvNet_V1(nb_classes=divide_class, Chans=channels, Samples=fs,
                           dropoutRate=0.5)
    model_name = "AeentionNet"
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

###############################################################################
# if the classification task was imbalanced (significantly more trials in one
# class versus the others) you can assign a weight to each class during
# optimization to balance it out. This data is approximately balanced so we
# don't need to do this, but is shown here for illustration/completeness.
###############################################################################

# the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
# the weights all to be 1
class_weights = {0: 1, 1: 1}

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

file_path = "./results"
f = open(file_path, 'a')
f.write("##############" + model_name + "#############" + '\n' + "subject:" + subject + '\n'
        + "epochs:" + str(epochs) + "\n" + "batch_size:" + str(batch_size) + "\n")
f.close()

loss=fittedModel.history['loss']
ax1=plt.subplot()
ax1.set_xlabel("iteration")
ax1.set_ylabel("training_loss")
plt.plot(loss,label='Training loss')
plt.title(model_name+" training loss")
plt.legend(loc=0,ncol=1)
plt.show()