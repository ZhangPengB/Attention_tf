import mne
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from mne import io


samplesfile = scipy.io.loadmat('./eeg_record1.mat')  # 文件读入字典
samples = samplesfile['o']['data'][0, 0].T # 提取字典中的numpy数组
print(samples)
print(samples.shape)



ch_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
            '20', '21', '22', '23', '24', '25']  # 通道名称
sfreq = 128  # 采样率
info = mne.create_info(ch_names, sfreq)  # 创建信号的信息
raw = mne.io.RawArray(samples, info)
print(type(raw))
# raw.save('atten_raw.fif')

# raw.plot()
# raw.show()
print('数据集的形状为：', raw.get_data().shape)
print('通道数为：', raw.info.get('nchan'))


raw_fname = 'atten_raw.fif'
# event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0., 1
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)

# Setup for reading the raw data
raw = io.Raw(raw_fname, preload=True, verbose=False)
print(raw)