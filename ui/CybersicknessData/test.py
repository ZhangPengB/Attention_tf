import os
from braindecode.models.eegnet import EEGNetv4, EEGNet
from braindecode.datautil.signalproc import highpass_cnt, bandpass_cnt


def load_data(path, _class):
    class_path = path + _class  # + "180" + ".mat"
    dirs = os.listdir(class_path)
    return dirs


dirs = load_data("E:/PyCharmProject/venv/egg/CybersicknessData/t2/", 'cyber')
print("dirs=", dirs)


