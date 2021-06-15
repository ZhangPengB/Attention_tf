from taskEngagement.EEGModels import EEGNet, ShallowConvNet, DeepConvNet
from taskEngagement.utils import produce_label, produce_data, write_predicition_message,produce_label_train_test,produce_data_train_test
from taskEngagement.ACNN import DeepConvNet_V1
from pyriemann.utils.viz import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Model

epoch, fs, channels = 1, 128, 64
sample_per = epoch * fs  # 每个样本包含的采样点数
names = ['moderate', 'low', 'high']
eegnet_flag = 3 # 加载模型标志

file_path = "./results"
f = open(file_path, 'a')

subject = 'zhaozhuren'
dirs = './datasets/' + subject + '/'
matfiles = ['zhaoqifei.mat', 'zhaojiangluo0.mat', 'zhaojiangluo1.mat']
train_data, valid_data, test_data = produce_data_train_test(dirs, matfiles)
# train_data, valid_data, test_data = produce_data(dirs, matfiles)

print(train_data.shape)
print(valid_data.shape)
print(test_data.shape)

y_train, y_valid, y_test = produce_label_train_test()
# y_train, y_valid, y_test = produce_label()
print(y_train.shape)

# model_name=""

if eegnet_flag == 1:
    model = EEGNet(nb_classes=3, Chans=channels, Samples=fs,
                   dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16,
                   dropoutType='Dropout')
    model_name = "eegnet"
    filepath = '/tmp/eegnet_checkPoints/'+subject+'/checkpoint.h5'
elif eegnet_flag == 0:
    model = ShallowConvNet(nb_classes=3, Chans=channels, Samples=fs,
                           dropoutRate=0.5)
    model_name = "shallowConvNet"
    filepath = '/tmp/shallowConvnet_checkPoints/'+subject+'/checkpoint.h5'
elif eegnet_flag == 2:
    model = DeepConvNet(nb_classes=3, Chans=64, Samples=128,
                        dropoutRate=0.5)
    model_name = "DeepConvNet"
    filepath = '/tmp/DeepConvnet_checkPoints/'+subject+'/checkpoint.h5'
else:
    model = DeepConvNet_V1(nb_classes=3, Chans=64, Samples=128,
                           dropoutRate=0.5)
    model_name = "AeentionNet"
    filepath = '/tmp/DeepConvnetV1_checkPoints/'+subject+'/checkpoint.h5'

print('----------开始预测-------------')
# load optimal weights
model.load_weights(filepath=filepath)
# Model.load_weights(filepath=filepath)
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
train_acc = np.mean(preds == y_train.argmax(axis=-1))
print("Classification accuracy train: %f " % (train_acc))
# write_predicition_message("./results", "train_accu:"+str(train_acc))
f.write("train_accu:" + str(train_acc) + '\n')

probs = model.predict(valid_data)
# print(probs[268:389])
preds_valid = probs.argmax(axis=-1)
print(preds_valid[200:240])
print(y_valid.argmax(axis=-1)[200:240])
valid_acc = np.mean(preds_valid == y_valid.argmax(axis=-1))
print("Classification accuracy valid: %f " % (valid_acc))
# write_predicition_message("./results", "valid_accu:"+str(valid_acc))
f.write("valid_accu:" + str(valid_acc) + "\n")

print(test_data[0].shape)

probs1 = model.predict(test_data[0].reshape(1,test_data[0].shape[0],test_data[0].shape[1],test_data[0].shape[2]))
preds1 = probs1.argmax(axis=-1)
print(preds1)

probs = model.predict(test_data)
# print(probs[268:389])
preds = probs.argmax(axis=-1)
print(preds[:60])
print(y_test.argmax(axis=-1)[200:240])
acc = np.mean(preds == y_test.argmax(axis=-1))
print("Classification accuracy  test: %f " % (acc))
# write_predicition_message("./results", "test_accu:"+str(acc))
f.write("test_accu:" + str(acc) + "\n")
f.close()

plt.figure(0)
plot_confusion_matrix(y_test.argmax(axis=-1), preds, names, title="test_"+model_name)
plt.show()
plot_confusion_matrix(y_valid.argmax(axis=-1), preds_valid, names, title="valid_"+model_name)
plt.show()
