import os
import numpy as np
from EEGModels_orig import EEGNet
from pyactivetwo import *
from keras.utils import np_utils
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# 训练模型
def train(server_port,ip_address,mask,nb_classes=2):
    train_data = get_data(server_port,ip_address,1)
    if mask == '0':
        train_label = np_utils.to_categorical(0,2)
    elif mask == '1':
        train_label = np_utils.to_categorical(1,2)
    else:
        raise ('error: no label')
    model = EEGNet(nb_classes,64,256)
    model.load_weights('checkpoint.h5')
    FittedModel = model.fit(train_data,train_label,batch_size=1,epochs=1,verbose=1)
    model.save('checkpoint.h5')

# 测试
def test(server_port,ip_address,nb_classes=3):

    test_data = get_data(server_port,ip_address,1)
    model = EEGNet(nb_classes,64,256)
    model.load_weights('checkpoint.h5')
    probs = model.predict(test_data)
    preds = np.argmax(probs, axis=-1)
    print("classification: %d " % (preds))
    return preds

# 打开exe程序
# os.startfile("D:\BaiduNetdisk\BaiduNetdisk.exe")
def open_exe(path):
    os.startfile(path)

# 写字符串到一个txt文件
def write_index(str_list):
    with open("./video.txt", "w", encoding='utf-8') as f:
        for name in str_list:
            f.write(name + '\n')
# open_exe('E:\Emotion_final\EyeLighting1.exe')

# global io
# try:
#     io = windll.LoadLibrary('inpoutx64.dll') # require dlportio.dll
# except:
#     print('The parallel port couldn\'t be opened!')

# def writeTrigger(lpt_port, trigger):
#     global io
#     try:
#         io.Out32(lpt_port, trigger)
#     except:
#         print('Failed to send trigger!')