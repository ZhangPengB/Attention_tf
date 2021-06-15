import pandas as pd
from pyriemann.utils.viz import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os

from utils import produce_data, produce_label
from tensorflow.keras.callbacks import ModelCheckpoint
from EEGModels import EEGNet, ShallowConvNet, DeepConvNet

epoch, fs, channels = 1, 128, 64
kernels = 1
sample_per = epoch * fs  # 每个样本包含的采样点数

# train
dirs = './datasets/'
name = 'zhangtrain_X3'
train_data = produce_data(dirs, name, epoch, fs, channels)
print(train_data.shape)

# valid
name = 'zhangvalid_X3'
valid_data = produce_data(dirs, name, epoch, fs, channels)
print(valid_data.shape)

# test
name = 'zhangtest_X3'
test_data = produce_data(dirs, name, epoch, fs, channels)
print(test_data.shape)

name = 'y3_train'
y_train = produce_label(dirs, name)
print(y_train.shape)

name = 'y3_valid'
y_valid = produce_label(dirs, name)
print(y_valid.shape)

name = 'y3_test'
y_test = produce_label(dirs, name)
print(y_test.shape)
print(y_test[:100])
print(y_test.argmax(axis=-1)[10:100])

print(y_test.argmax(axis=-1)[268:360])


model = EEGNet(nb_classes=2, Chans=channels, Samples=fs,
               dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16,
               dropoutType='Dropout')

# model = ShallowConvNet(nb_classes=3, Chans=channels, Samples=fs,
#                dropoutRate=0.5)
# model = DeepConvNet(nb_classes=3, Chans=64, Samples=128,
#                     dropoutRate=0.5)

# compile the model and set the optimizers
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# count number of parameters in the model
numParams = model.count_params()

# set a valid path for your system to record model checkpoints
checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                               save_best_only=True)
print("----------dene-------")

class_weights = {0: 1, 1: 1, 2: 1}

################################################################################
# fit the model. Due to very small sample sizes this can get
# pretty noisy run-to-run, but most runs should be comparable to xDAWN +
# Riemannian geometry classification (below)
################################################################################
fittedModel = model.fit(train_data, y_train, batch_size=8, epochs=200,
                        verbose=2, validation_data=(valid_data, y_valid),
                        callbacks=[checkpointer], class_weight=class_weights)

# load optimal weights
model.load_weights('/tmp/checkpoint.h5')

##############################################################################
# can alternatively used the weights provided in the repo. If so it should get
# you 93% accuracy. Change the WEIGHTS_PATH variable to wherever it is on your
# system.
###############################################################################

# WEIGHTS_PATH = /path/to/EEGNet-8-2-weights.h5
# model.load_weights(WEIGHTS_PATH)

probs = model.predict(train_data)
# print(probs[268:389])
preds = probs.argmax(axis=-1)
print(preds[600:700])
print(y_train.argmax(axis=-1)[600:700])
acc = np.mean(preds == y_train.argmax(axis=-1))
print("Classification accuracy train: %f " % (acc))

probs = model.predict(valid_data)
# print(probs[268:389])
preds = probs.argmax(axis=-1)
print(preds[200:240])
print(y_valid.argmax(axis=-1)[200:240])
acc = np.mean(preds == y_valid.argmax(axis=-1))
print("Classification accuracy valid: %f " % (acc))

probs = model.predict(test_data)
# print(probs[268:389])
preds = probs.argmax(axis=-1)
print(preds[200:240])
print(y_test.argmax(axis=-1)[200:240])
acc = np.mean(preds == y_test.argmax(axis=-1))
print("Classification accuracy  test: %f " % (acc))

names = ['moderate', 'low', 'high']
plt.figure(0)
plot_confusion_matrix(preds, y_test.argmax(axis=-1), names, title='ShallowConvNet')
plt.show()
