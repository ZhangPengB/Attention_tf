from OnlineAttention.ACNN import ACNN
import numpy as np
from tensorflow.keras import Model
from OnlineAttention.get_DataOnline import loaddata_online


subject = 'zhaozhuren'
model = ACNN(nb_classes=3, Chans=64, Samples=128,
             dropoutRate=0.5)
model_name = "AeentionNet"
filepath = '/tmp/DeepConvnetV1_checkPoints/' + subject + '/checkpoint.h5'


if __name__ == '__main__':
    print('----------开始在线预测-------------')
    # load optimal weights
    model.load_weights(filepath=filepath)

    # load test data
    test_data = loaddata_online()
    print(test_data.shape)

    probs = model.predict(test_data)
    preds = probs.argmax(axis=-1)
    print("preds:",preds)
    if preds == 2:
        print("medium attention")
    elif preds == 1:
        print("low attention")
    else:
        print("high attention")

