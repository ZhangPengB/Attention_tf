import scipy.io
import time
import numpy as np
from OnlineAttention.get_data import ActiveTwo
from braindecode.datautil.signalproc import highpass_cnt, bandpass_cnt, exponential_running_demean, \
    exponential_running_standardize
# from braindecode.datautil import exponential_moving_demean,exponential_moving_standardize
from OnlineAttention.ACNN import ACNN


def loaddata_online():
    host = '10.127.0.1'
    sfreq = 1024
    port = 1111
    channle_num = 65
    duration = 1

    def dim22dim3(dim2_data):
        l = 128
        time_len = dim2_data.shape[1] // l
        dim3_data = np.zeros((time_len, 64, l)).astype(np.float32)
        for i in range(time_len):
            dim3_data[i] = dim2_data[:, l * i:l + l * i]
        return dim3_data

    active_two = ActiveTwo(host=host, sfreq=sfreq, port=port, nchannels=channle_num)
    raw_data = active_two.read(duration=duration)
    data = raw_data[0:64, :]
    # np.save('data', np.array(data))
    l = data.shape[1]
    tmp = data[:, range(0, l, 8)]
    tmp = exponential_running_demean(tmp)
    tmp = exponential_running_standardize(tmp)
    # tmp=exponential_moving_demean(tmp)
    # tmp=exponential_moving_standardize(tmp)
    tmp = bandpass_cnt(tmp, 0.1, 40, 128, filt_order=3, axis=1)
    data = dim22dim3(tmp)
    # data = np.reshape(data, (data.shape[0], 1, data.shape[1], data.shape[2]))
    data = np.reshape(data, (1, data.shape[0], data.shape[1], data.shape[2]))
    # data_loader = DataLoader(data, batch_size=1)
    return data



if __name__ == '__main__':
    subject = 'zhaozhuren'
    model = ACNN(nb_classes=3, Chans=64, Samples=128,
                 dropoutRate=0.5)
    model_name = "AeentionNet"
    filepath = '/tmp/DeepConvnetV1_checkPoints/' + subject + '/checkpoint.h5'
    # load optimal weights
    model.load_weights(filepath=filepath)

    print('----------开始在线预测-------------')
    while True:
        # load test data
        test_data = loaddata_online()
        print(test_data.shape)

        probs = model.predict(test_data)
        preds = probs.argmax(axis=-1)
        print("preds:", preds)
        if preds == 2:
            print("medium attention")
        elif preds == 1:
            print("low attention")
        else:
            print("high attention")
