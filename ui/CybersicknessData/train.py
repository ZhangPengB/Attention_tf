import os
import numpy as np
import scipy.io as sco
import mne
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from numpy.random import RandomState
import random
import torch as th
import torch.nn.functional as F

#  /home/frost/anaconda3/envs/py36/lib/python3.6/site-packages
from braindecode.torch_ext.util import set_random_seeds
from braindecode.torch_ext.util import np_to_var, var_to_np
from braindecode.torch_ext.optimizers import AdamW
from braindecode.torch_ext.schedulers import ScheduledOptimizer, CosineAnnealing
from braindecode.models.eegnet import EEGNetv4, EEGNet
from braindecode.models.util import to_dense_prediction_model
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.experiments.monitors import compute_preds_per_trial_from_crops
from braindecode.datautil.signalproc import highpass_cnt, bandpass_cnt
import braindecode.models.deep4


# from braindecode.datautil.preproc

# from cc import EEGNetv4c
# from EEG_n import EEGNet_new
# from EEGNet import EEGNet_torch

import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *





class WorkThread(QThread):

    resultList = []
    testList = []
    printResult = pyqtSignal()
    end = pyqtSignal()
    def sendSignal(self, result=[], params='Epoch', is_print = False):
        """
        :param result: output to the text bowser
        :param params: what type of the param is
            'Epoch' 'Loss' 'Tloss' 'Acc' 'Tloss' 'Predicted' 'Label' 'Tacc'
        :return:
        """
        self.param = params
        if self.param == 'Epoch':
            # self.result = "Epoch {:d}".format(result)
            self.resultList.append("Epoch {:d}".format(result))
        elif self.param == 'Loss':
            self.resultList.append("{:6s} Loss: {:.5f}".format(result[0], result[1]))
            # self.result = "{:6s} Loss: {:.5f}".format(result[0], result[1])
        elif self.param == 'Acc':
            self.resultList.append("{:6s} Accuracy: {:.1f}%".format(result[0], result[1] * 100))
            # self.result = "{:6s} Accuracy: {:.1f}%".format(result[0], result[1] * 100)
        elif self.param == 'Tloss':
            self.testList.append("Test Loss: {:.5f}".format(result))
            # self.result = "Test Loss: {:.5f}".format(result)
        elif self.param == 'Predicted':
            self.testList.append(' '.join([str(i) for i in result.tolist()]))
        elif self.param == 'Label':
            self.testList.append(' '.join([str(i) for i in result.tolist()]))
        elif self.param == 'Tacc':
            self.testList.append("Test Accuracy: {:.1f}%\n".format(result * 100))
            # self.result = "Test Accuracy: {:.1f}%\n".format(result * 100)
        if is_print:
            self.result = "\n".join(self.resultList)
            self.resultList = []
            self.printResult.emit()
        if self.param == 'Tacc':
            self.testResult = "\n".join(self.testList)
            self.end.emit()


    def run(self):
        path = "./t2/"

        from load_data import load_data, dim22dim3, train_data, test_data, valid_data

        cyber_data = load_data(path, "cyber")
        silent_data = load_data(path, "silent")

        cyber_train_data = train_data(cyber_data)
        cyber_valid_data = valid_data(cyber_data)
        cyber_test_data = test_data(cyber_data)
        silent_train_data = train_data(silent_data)
        silent_valid_data = valid_data(silent_data)
        silent_test_data = test_data(silent_data)

        train_data = np.hstack((cyber_train_data, silent_train_data))  # train_data.shape=(64,221184)
        # print("train_data.shape=",train_data.shape)
        valid_data = np.hstack((cyber_valid_data, silent_valid_data))  # valid_data.shape=(64,73728)
        # print("valid_data.shape=",valid_data.shape)
        test_data = np.hstack((cyber_test_data, silent_test_data))  # test_data.shape=(64,73728)
        # print("test_data.shape=",test_data.shape)

        train_data = bandpass_cnt(train_data, 14, 49, 1024, filt_order=3, axis=1)  # train_data.shape=(64,221184)
        valid_data = bandpass_cnt(valid_data, 14, 49, 1024, filt_order=3, axis=1)  # valid_data.shape=(64,73728)
        test_data = bandpass_cnt(test_data, 14, 49, 1024, filt_order=3, axis=1)  # test_data.shape=(64,73728)

        # train_data = bandpass_cnt(train_data, 10, 50, 1024, filt_order=3, axis=1)
        # valid_data = bandpass_cnt(valid_data, 10, 50, 1024, filt_order=3, axis=1)
        # test_data = bandpass_cnt(test_data, 10, 50, 1024, filt_order=3, axis=1)

        train_data = dim22dim3(train_data)  # train_data.shape=(216,64,1024)
        valid_data = dim22dim3(valid_data)  # valid_data.shape=(72,64,1024)
        test_data = dim22dim3(test_data)  # test_data.shape=(72,64,1024)

        s2 = 1
        train_label = np.hstack((np.zeros(108 * s2) + 1,
                                 np.zeros(108 * s2)))  # train_label.shape=(216,)
        valid_label = np.hstack((np.zeros(36 * s2) + 1,
                                 np.zeros(36 * s2)))  # valid_label.shape=(72,)
        test_label = np.hstack((np.zeros(36 * s2) + 1,
                                np.zeros(36 * s2)))  # test_label.shape=(72,)

        # cyber_data = load_data(path, "cyber")
        # silent_data = load_data(path, "silent")
        #
        # cyber_data = bandpass_cnt(cyber_data, 14, 49, 1024, filt_order=3, axis=1)
        # silent_data = bandpass_cnt(silent_data, 14, 49, 1024, filt_order=3, axis=1)
        #
        # cyber_len = cyber_data.shape[1] // 1024
        # silent_len = silent_data.shape[1] // 1024
        # c1 = int(cyber_len*0.6)
        # c2 = int(cyber_len*0.8)
        # s1 = int(silent_len*0.6)
        # s2 = int(silent_len*0.8)
        #
        # train_data = dim22dim3(np.hstack((cyber_data[:, :c1*1024],
        #                                   silent_data[:, :s1*1024])))
        # valid_data = dim22dim3(np.hstack((cyber_data[:, c1*1024 : c2*1024],
        #                                   silent_data[:, s1*1024 : s2*1024])))
        # test_data = dim22dim3(np.hstack((cyber_data[:, c2*1024:],
        #                                  silent_data[:, s2*1024:])))

        ## cyber 1   silent 0
        # train_label = np.hstack((np.zeros(c1) + 1,
        #                          np.zeros(s1)))
        # valid_label = np.hstack((np.zeros(c2 - c1) + 1,
        #                          np.zeros(s2 - s1)))
        # test_label = np.hstack((np.zeros(cyber_len - c2) + 1,
        #                         np.zeros(silent_len - c2)))

        train_set = SignalAndTarget(train_data, train_label)  # train_set.type=SignalAndTarget with (X,y)
        valid_set = SignalAndTarget(valid_data, valid_label)  # valid_set.type=SignalAndTarget with (X,y)
        test_set = SignalAndTarget(test_data, test_label)  # test_set.type=SignalAndTarget  with (X,y)

        # Set if you want to use GPU
        # You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
        cuda = True
        set_random_seeds(seed=20181019, cuda=cuda)
        n_classes = 2
        in_chans = train_set.X.shape[1]  # in_chans=64
        # final_conv_length = auto ensures we only get a single output in the time dimension

        # model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes,
        #                         input_time_length=train_set.X.shape[2],
        #                         final_conv_length=12).create_network()

        model = EEGNetv4(in_chans=in_chans, n_classes=n_classes,
                         input_time_length=train_set.X.shape[2],
                         final_conv_length=12).create_network()

        # model = EEGNetv4c(in_chans=in_chans, n_classes=n_classes,
        #                  input_time_length=train_set.X.shape[2],
        #                  final_conv_length=12).create_network()

        # model = EEGNet_new(in_chans=in_chans, n_classes=n_classes,
        #                  input_time_length=train_set.X.shape[2],
        #                  final_conv_length=12).create_network()

        to_dense_prediction_model(model)  # 将跨步模型改为输出密集型模型

        if cuda:
            model.cuda()
        model.eval()

        # convert ndarray to torch.tensor
        test_input = np_to_var(np.ones((2, in_chans, train_set.X.shape[2], 1), dtype=np.float32))  # test_input=torch.tensor

        if cuda:
            test_input = test_input.cuda()
        out = model(test_input)

        n_preds_per_input = out.cpu().data.numpy().shape[2]
        iterator = CropsFromTrialsIterator(batch_size=16, input_time_length=train_set.X.shape[2],
                                           n_preds_per_input=n_preds_per_input)

        # rng = RandomState((2018,8,7))

        optimizer = AdamW(model.parameters(), lr=1 * 0.01,
                          weight_decay=0.5 * 0.001)  # these are good values for the deep model
        # optimizer = AdamW(model.parameters(), lr=0.00001, weight_decay=0.5 * 0.00001)
        # Need to determine number of batch passes per epoch for cosine annealing
        n_epochs = 10
        n_updates_per_epoch = len([None for b in iterator.get_batches(train_set, True)])
        scheduler = CosineAnnealing(n_epochs * n_updates_per_epoch)
        # schedule_weight_decay must be True for AdamW
        optimizer = ScheduledOptimizer(scheduler, optimizer, schedule_weight_decay=True)

        rng = RandomState((2017, 6, 30))
        for i_epoch in range(n_epochs):
            # Set model to training mode
            model.train()
            for batch_X, batch_y in iterator.get_batches(train_set, shuffle=True):
                net_in = np_to_var(batch_X)
                if cuda:
                    net_in = net_in.cuda()
                net_target = np_to_var(batch_y).long()
                if cuda:
                    net_target = net_target.cuda()
                # Remove gradients of last backward pass from all parameters
                optimizer.zero_grad()
                outputs = model(net_in)
                # Mean predictions across trial
                # Note that this will give identical gradients to computing
                # a per-prediction loss (at least for the combination of log softmax activation
                # and negative log likelihood loss which we are using here)
                outputs = th.mean(outputs, dim=2, keepdim=False)
                loss = F.nll_loss(outputs, net_target)
                loss.backward()
                optimizer.step()

            # Print some statistics each epoch
            model.eval()
            self.sendSignal(i_epoch, 'Epoch')
            print("Epoch {:d}".format(i_epoch))
            for setname, dataset in (('Train', train_set), ('Valid', valid_set)):
                # Collect all predictions and losses
                all_preds = []
                all_losses = []
                batch_sizes = []
                for batch_X, batch_y in iterator.get_batches(dataset, shuffle=False):
                    net_in = np_to_var(batch_X)
                    if cuda:
                        net_in = net_in.cuda()
                    net_target = np_to_var(batch_y).long()
                    if cuda:
                        net_target = net_target.cuda()
                    outputs = model(net_in)
                    all_preds.append(var_to_np(outputs))
                    outputs = th.mean(outputs, dim=2, keepdim=False)
                    loss = F.nll_loss(outputs, net_target)
                    loss = float(var_to_np(loss))
                    all_losses.append(loss)
                    batch_sizes.append(len(batch_X))
                # Compute mean per-input loss
                loss = np.mean(np.array(all_losses) * np.array(batch_sizes) /
                               np.mean(batch_sizes))
                self.sendSignal([setname, loss], 'Loss')
                print("{:6s} Loss: {:.5f}".format(setname, loss))
                # Assign the predictions to the trials
                preds_per_trial = compute_preds_per_trial_from_crops(all_preds,
                                                                     train_set.X.shape[2],
                                                                     dataset.X)
                # preds per trial are now trials x classes x timesteps/predictions
                # Now mean across timesteps for each trial to get per-trial predictions
                meaned_preds_per_trial = np.array([np.mean(p, axis=1) for p in preds_per_trial])
                predicted_labels = np.argmax(meaned_preds_per_trial, axis=1)
                accuracy = np.mean(predicted_labels == dataset.y)
                self.sendSignal([setname, accuracy], 'Acc')
                print("{:6s} Accuracy: {:.1f}%".format(
                    setname, accuracy * 100))
            self.sendSignal(None, None, is_print=True)

        model.eval()
        # Collect all predictions and losses
        all_preds = []
        all_losses = []
        batch_sizes = []
        for batch_X, batch_y in iterator.get_batches(test_set, shuffle=False):
            net_in = np_to_var(batch_X)
            if cuda:
                net_in = net_in.cuda()
            net_target = np_to_var(batch_y).long()
            if cuda:
                net_target = net_target.cuda()
            outputs = model(net_in)
            look = outputs
            all_preds.append(var_to_np(outputs))
            outputs = th.mean(outputs, dim=2, keepdim=False)
            loss = F.nll_loss(outputs, net_target)
            loss = float(var_to_np(loss))
            all_losses.append(loss)
            batch_sizes.append(len(batch_X))
        # Compute mean per-input loss
        loss = np.mean(np.array(all_losses) * np.array(batch_sizes) /
                       np.mean(batch_sizes))
        print("Test Loss: {:.5f}".format(loss))
        # Assign the predictions to the trials
        preds_per_trial = compute_preds_per_trial_from_crops(all_preds,
                                                             test_set.X.shape[2],
                                                             test_set.X)
        # preds per trial are now trials x classes x timesteps/predictions
        # Now mean across timesteps for each trial to get per-trial predictions
        meaned_preds_per_trial = np.array([np.mean(p, axis=1) for p in preds_per_trial])
        predicted_labels = np.argmax(meaned_preds_per_trial, axis=1)
        accuracy = np.mean(predicted_labels == test_set.y)
        half_num = len(predicted_labels) // 2
        print(predicted_labels[:half_num])
        print(predicted_labels[half_num:])
        print("Test Accuracy: {:.1f}%".format(accuracy * 100))
        self.sendSignal(loss, 'Tloss')
        self.sendSignal(predicted_labels[:half_num], 'Predicted')
        self.sendSignal(predicted_labels[half_num:], 'Label')
        self.sendSignal(accuracy, 'Tacc')

class QW(QWidget):
    def __init__(self, parent=None):
        super(QW, self).__init__(parent)
        self.setWindowTitle("Cyber Sickness输出结果")
        self.resize(500,300)


        layout = QVBoxLayout()
        self.textBrowser = QTextBrowser()
        self.textBrowser.setFontPointSize(20)
        # self.textBrowser.setFont(QFont=QFont("Yahei"))
        layout.addWidget(self.textBrowser)
        self.button = QPushButton('start training')
        layout.addWidget(self.button)
        self.workThread = WorkThread()
        self.workThread.printResult.connect(self.TextAppend)
        self.workThread.end.connect(self.End)
        self.button.clicked.connect(self.Work)
        self.setLayout(layout)



    def TextAppend(self):
        self.textBrowser.clear()
        self.textBrowser.append(self.workThread.result)
        self.textBrowser.moveCursor(self.textBrowser.textCursor().End)

    def Work(self):
        self.workThread.start()

    def End(self):
        QMessageBox.information(self, '测试结果', self.workThread.testResult, QMessageBox.Ok)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = QW()
    form.show()
    sys.exit(app.exec_())





