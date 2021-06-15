import scipy.io
import time
import numpy as np
import torch

from OnlineAttention.get_data import ActiveTwo
from braindecode.datautil.signalproc import highpass_cnt, bandpass_cnt, exponential_running_demean, exponential_running_standardize


type = 'low'
num = str(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = '..\\rawdata\\%(name)s.mat' % {'name': type+num}
mat = scipy.io.loadmat(path)['{0}'.format(type)]


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
    tmp = bandpass_cnt(tmp, 0.1, 40, 128, filt_order=3, axis=1)
    data = dim22dim3(tmp)
    data = np.reshape(data, (data.shape[0], 1, data.shape[1], data.shape[2]))
    data_loader = DataLoader(data, batch_size=1)
    return data_loader


def loaddata(mat, start):
    def dim22dim3(dim2_data):
        l = 128
        time_len = dim2_data.shape[1] // l
        dim3_data = np.zeros((time_len, 64, l)).astype(np.float32)
        for i in range(time_len):
            dim3_data[i] = dim2_data[:, l * i:l + l * i]
        return dim3_data

    t = 1024
    data = mat[:, start*t: start*t+1024]
    l = data.shape[1]
    tmp1 = data[:, range(0, l, 8)]
    tmp2 = data[:, range(1, l, 8)]
    tmp3 = data[:, range(2, l, 8)]
    tmp4 = data[:, range(3, l, 8)]
    tmp5 = data[:, range(4, l, 8)]
    tmp6 = data[:, range(5, l, 8)]
    tmp7 = data[:, range(6, l, 8)]
    tmp8 = data[:, range(7, l, 8)]
    tmp = np.hstack((tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8))
    tmp = exponential_running_demean(tmp)
    tmp = exponential_running_standardize(tmp)
    tmp = bandpass_cnt(tmp, 0.1, 40, 128, filt_order=3, axis=1)

    data = dim22dim3(tmp)
    data = np.reshape(data, (data.shape[0], 1, data.shape[1], data.shape[2]))  # 576个样本点,1,64,64
    # class MyDataset(Dataset):
    #     def __init__(self, test):
    #         super(MyDataset, self).__init__()
    #         self.transform = transforms.ToTensor()
    #         self.x = test
    #
    #     def __getitem__(self, index):
    #         x = self.x[index, :, :, :]
    #         # x, y = self.transform(x), self.transform(y)
    #         return x
    #
    #     def __len__(self):
    #         return len(self.x)

    # data = MyDataset(data)

    data_loader = DataLoader(data, batch_size=1)
    return data_loader



def test(model, data_loader, device):
    model.eval()
    pre_label = []

    for index, data in enumerate(data_loader):
        x = data
        x = x.to(device)

        y_pred, _ = model(x)
        _, pred = torch.max(y_pred, 1)  # values, indices

        for i in pred.cpu().numpy():
            pre_label.append(i)

    # print(len(pre_label))
    # print(pre_label)
    return pre_label


PATH = '..\\model\\test15.pkl'
model = torch.load(PATH)
model = model.to(DEVICE)


def analysis():
    tmp = []
    for i in range(108, 180):
        data_loader = loaddata(mat, i)
        print(i+1)
        label = test(model, data_loader, DEVICE)
        tmp.append(sum(label)//8)
        print(label)
        time.sleep(1)


    # print(tmp)
    # tmp1, tmp2, tmp3, tmp4 = 0, 0, 0, 0
    # for i in tmp:
    #     if i == 0:
    #         tmp1 += 1
    #     elif i == 1:
    #         tmp2 += 1
    #     elif i == 2:
    #         tmp3 += 1
    #     else:
    #         tmp4 += 1
    # print(tmp1, tmp2, tmp3, tmp4)

    return


def analysis_online():
    data_loader = loaddata_online()
    label = test(model, data_loader, DEVICE)
    print(label)


if __name__ == '__main__':
    analysis()


