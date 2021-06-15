from KDTest.PreHandleData import produce_label, produce_data_train_test
from KDTest.EEGModels import EEGNet, ShallowConvNet, DeepConvNet
from KDTest.ACNN import DeepConvNet_V1
from pyriemann.utils.viz import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Model

divide_class=2
sample_len, fs, channels = 1, 128, 64
divide=0.8   #数据划分比例
timelen=10
names = ['extream_attention','general_attention']
model_flag = 1 # 加载模型标志


file_path = "./results"
f = open(file_path, 'a')

# 加载数据
subject = 'wuailin'
dirs = './datasets/' + subject + '/'
matfiles = ['wu_P_1.mat', 'wu_N_1.mat']

train_data, valid_data, test_data = produce_data_train_test(dirs, matfiles, timelen=timelen, divide=divide)

print(train_data.shape)
print(valid_data.shape)
print(test_data.shape)

y_train, y_valid, y_test = produce_label(timelen=timelen,divide=divide)

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



print('----------开始预测-------------')
# load optimal weights
model.load_weights(filepath=filepath)


preds = model.predict(train_data).argmax(axis=-1)
print(y_train.argmax(axis=-1)[600:700])
train_acc = np.mean(preds == y_train.argmax(axis=-1))
print("Classification accuracy of train: %f " % (train_acc))
f.write("train_accu:" + str(train_acc) + '\n')

preds_valid = model.predict(valid_data).argmax(axis=-1)
print(preds_valid[200:240])
print(y_valid.argmax(axis=-1)[200:240])
valid_acc = np.mean(preds_valid == y_valid.argmax(axis=-1))
print("Classification accuracy of valid: %f " % (valid_acc))
f.write("valid_accu:" + str(valid_acc) + "\n")

# print(test_data[0].shape)
#
# probs1 = model.predict(test_data[0].reshape(1,test_data[0].shape[0],test_data[0].shape[1],test_data[0].shape[2]))
# preds1 = probs1.argmax(axis=-1)
# print(preds1)

preds = model.predict(test_data).argmax(axis=-1)
print(preds[:60])
print(y_test.argmax(axis=-1)[200:240])
acc = np.mean(preds == y_test.argmax(axis=-1))
print("Classification accuracy of test: %f " % (acc))
f.write("test_accu:" + str(acc) + "\n")
f.close()

plt.figure(0)
plot_confusion_matrix(y_test.argmax(axis=-1), preds, names, title="test_"+model_name)
plt.show()
plot_confusion_matrix(y_valid.argmax(axis=-1), preds_valid, names, title="valid_"+model_name)
plt.show()
