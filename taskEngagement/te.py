import numpy as np

from utils import produce_data, produce_label
from tensorflow.keras.callbacks import ModelCheckpoint
from EEGModels import EEGNet

epoch, fs, channels = 1, 1024, 64
kernels = 1

dirs = './datasets/'
name = 'train'
train_data = produce_data(dirs, name, epoch, fs, channels)
print(train_data.shape)  # (1934, 1, 64, 1024)


# valid
name = 'valid'
valid_data = produce_data(dirs, name, epoch, fs, channels)
print(valid_data.shape)  #(645, 1, 64, 1024)

# test
name = 'test'
test_data = produce_data(dirs, name, epoch, fs, channels)
print(test_data.shape)  #(645, 1, 64, 1024)

name = 'y_train_nosample'
y_train = produce_label(dirs, name)
print(y_train.shape)

name = 'y_valid'
y_valid = produce_label(dirs, name)
print(y_valid.shape)

name = 'y_test'
y_test = produce_label(dirs, name)
print(y_test.shape)


model = EEGNet(nb_classes=3, Chans=channels, Samples=fs,
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

class_weights = {0: 1, 1: 1, 2: 1}

################################################################################
# fit the model. Due to very small sample sizes this can get
# pretty noisy run-to-run, but most runs should be comparable to xDAWN +
# Riemannian geometry classification (below)
################################################################################
fittedModel = model.fit(train_data, y_train, batch_size=8, epochs=150,
                        verbose=2, validation_data=(valid_data, y_valid),
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

probs = model.predict(train_data)
# print(probs[268:389])
preds = probs.argmax(axis=-1)
print(preds[803:1168])
print(y_train.argmax(axis=-1)[803:1168])
acc = np.mean(preds == y_train.argmax(axis=-1))
print("Classification accuracy train: %f " % (acc))

probs = model.predict(valid_data)
# print(probs[268:389])
preds = probs.argmax(axis=-1)
print(preds[268:389])
print(y_valid.argmax(axis=-1)[268:389])
acc = np.mean(preds == y_valid.argmax(axis=-1))
print("Classification accuracy valid: %f " % (acc))

probs = model.predict(test_data)
# print(probs[268:389])
preds = probs.argmax(axis=-1)
print(preds[268:389])
print(y_test.argmax(axis=-1)[268:389])
acc = np.mean(preds == y_test.argmax(axis=-1))
print("Classification accuracy  test: %f " % (acc))