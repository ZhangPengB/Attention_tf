import sys
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QRect
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QMouseEvent
from Emotion_online.communication import *
from multiprocessing import Process
# from Emotion_online.utils import *
import numpy as np

from OnlineAttention.get_data import ActiveTwo

from OnlineAttention.ACNN import ACNN
from braindecode.datautil.signalproc import highpass_cnt, bandpass_cnt, exponential_running_demean, \
    exponential_running_standardize


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.mode = ''

    def initUI(self):

        self.setObjectName('MainWindow')
        self.resize(1440, 1000)
        self.setWindowFlags(Qt.FramelessWindowHint)  # 去边框
        self.setAttribute(Qt.WA_TranslucentBackground)  # 设置窗口背景透明

        self.background = QLabel(self)
        self.background.setGeometry(QRect(0, 0, 1440, 1000))
        self.background.setPixmap(QPixmap("./pic/background.png"))
        self.background.setScaledContents(True)
        self.background.setObjectName("background")

        self.title = QLabel(self)
        self.title.setGeometry(QRect(576, 120, 300, 60))
        self.title.setPixmap(QPixmap("./pic/title.png"))
        self.title.setScaledContents(True)
        self.title.setObjectName("title")

        # self.positiveHint = QLabel(self)
        # self.positiveHint.setGeometry(QRect(165, 200, 300, 60))
        # self.positiveHint.setPixmap(QPixmap("./pic/positiveHint.png"))
        # self.positiveHint.setScaledContents(True)
        # self.positiveHint.setObjectName("positiveHint")

        # self.negativeHint = QLabel(self)
        # self.negativeHint.setGeometry(QRect(985, 200, 300, 60))
        # self.negativeHint.setPixmap(QPixmap("./pic/negativeHint.png"))
        # self.negativeHint.setScaledContents(True)
        # self.negativeHint.setObjectName("negativeHint")

        self.present = QLabel(self)
        self.present.setGeometry(QRect(526, 287, 400, 327))
        self.present.setPixmap(QPixmap("./pic/gray.png"))
        self.present.setScaledContents(True)
        self.present.setObjectName("present")

        self.high = QLabel(self)
        self.high.setGeometry(QRect(526, 287, 400, 327))
        self.high.setPixmap(QPixmap("./pic/high.png"))
        self.high.setScaledContents(False)
        self.high.setObjectName("high")

        self.low = QLabel(self)
        self.low.setGeometry(QRect(526, 287, 400, 327))
        self.low.setPixmap(QPixmap("./pic/low.png"))
        self.low.setScaledContents(False)
        self.low.setObjectName("low")

        self.med = QLabel(self)
        self.med.setGeometry(QRect(526, 287, 400, 327))
        self.med.setPixmap(QPixmap("./pic/med.png"))
        self.med.setScaledContents(False)
        self.med.setObjectName("med")


        self.minButton = QPushButton(self)
        self.minButton.setGeometry(QRect(1290, 90, 16, 16))
        self.minButton.setStyleSheet("QPushButton{\n"
                                     "    background:#6C6C6C;\n"
                                     "    color:white;\n"
                                     "    box-shadow: 1px 1px 3px rgba(0,0,0,0.3);font-size:16px;border-radius: 8px;font-family: 微软雅黑;\n"
                                     "}\n"
                                     "QPushButton:hover{                    \n"
                                     "    background:#9D9D9D;\n"
                                     "}\n"
                                     "QPushButton:pressed{\n"
                                     "    border: 1px solid #3C3C3C!important;\n"
                                     "}")
        self.minButton.setObjectName("minButton")
        self.minButton.clicked.connect(self.min_on_click)

        self.exitButton = QPushButton(self)
        self.exitButton.setGeometry(QRect(1320, 90, 16, 16))
        self.exitButton.setStyleSheet("QPushButton{\n"
                                      "    background:#CE0000;\n"
                                      "    color:white;\n"
                                      "    box-shadow: 1px 1px 3px rgba(0,0,0,0.3);font-size:16px;border-radius: 8px;font-family: 微软雅黑;\n"
                                      "}\n"
                                      "QPushButton:hover{                    \n"
                                      "    background:#FF2D2D;\n"
                                      "}\n"
                                      "QPushButton:pressed{\n"
                                      "    border: 1px solid #3C3C3C!important;\n"
                                      "    background:#AE0000;\n"
                                      "}")
        self.exitButton.setObjectName("exitButton")
        self.exitButton.clicked.connect(self.exit_on_click)

        self.grid = QGridLayout(self)
        for index in range(10):
            checkbox = QCheckBox(self)
            if index % 2 == 0:
                checkbox.setGeometry(QRect(225, 300 + 60 * (index // 2), 90, 40))
            else:
                checkbox.setGeometry(QRect(335, 300 + 60 * (index // 2), 90, 40))
            checkbox.setCheckable(True)
            checkbox.setStyleSheet("QCheckBox{\n"
                                   "    background:transparent;\n"
                                   "    color:black;\n"
                                   "    box-shadow: 1px 1px 1px rgba(0,0,0,0.3);font-size:20px;font-family: 微软雅黑;\n"
                                   "}\n"
                                   "QCheckBox::indicator{\n"
                                   "    width:30px;\n"
                                   "    height:30px;\n"
                                   "}\n"
                                   "QCheckBox::indicator:unchecked{\n"
                                   "    background:#EEE8CD;\n"
                                   "}\n"
                                   "QCheckBox::indicator:checked{\n"
                                   "    background:#BDD8F2;\n"
                                   "}\n"
                                   )
            checkbox.setObjectName("checkbox")
            checkbox.setText(QCoreApplication.translate("MainWindow", "P_" + str(index)))
            checkbox.stateChanged.connect(self.write_video)
            self.grid.addWidget(checkbox)

        for index in range(10, 20):
            checkbox = QCheckBox(self)
            if index % 2 == 0:
                checkbox.setGeometry(QRect(1045, 300 + 60 * ((index - 10) // 2), 90, 40))
            else:
                checkbox.setGeometry(QRect(1155, 300 + 60 * ((index - 10) // 2), 90, 40))
            checkbox.setCheckable(True)
            checkbox.setStyleSheet("QCheckBox{\n"
                                   "    background:transparent;\n"
                                   "    color:black;\n"
                                   "    box-shadow: 1px 1px 1px rgba(0,0,0,0.3);font-size:20px;font-family: 微软雅黑;\n"
                                   "}\n"
                                   "QCheckBox::indicator{\n"
                                   "    width:30px;\n"
                                   "    height:30px;\n"
                                   "}\n"
                                   "QCheckBox::indicator:unchecked{\n"
                                   "    background:#EEE8CD;\n"
                                   "}\n"
                                   "QCheckBox::indicator:checked{\n"
                                   "    background:#BDD8F2;\n"
                                   "}\n"
                                   )
            checkbox.setObjectName("checkbox")
            checkbox.setText(QCoreApplication.translate("MainWindow", "N_" + str(index - 10)))
            checkbox.stateChanged.connect(self.write_video)
            self.grid.addWidget(checkbox)

        self.ModeHint = QLabel(self)
        self.ModeHint.setGeometry(QRect(556, 640, 150, 30))
        self.ModeHint.setStyleSheet("font-family:微软雅黑;"
                                    "font-size:20px;")
        self.ModeHint.setObjectName("ModeHint")
        self.ModeHint.setText(QCoreApplication.translate("MainWindow", "Mode:"))
        self.ModeHint.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.serverHint = QLabel(self)
        self.serverHint.setGeometry(QRect(556, 690, 150, 30))
        self.serverHint.setStyleSheet("font-family:微软雅黑;"
                                      "font-size:20px;")
        self.serverHint.setObjectName("serverHint")
        self.serverHint.setText(QCoreApplication.translate("MainWindow", "serverPort:"))
        self.serverHint.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.ipAddressHint = QLabel(self)
        self.ipAddressHint.setGeometry(QRect(556, 740, 150, 30))
        self.ipAddressHint.setStyleSheet("font-family:微软雅黑;"
                                         "font-size:20px;")
        self.ipAddressHint.setObjectName("ipAddressHint")
        self.ipAddressHint.setText(QCoreApplication.translate("MainWindow", "ipAddress:"))
        self.ipAddressHint.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.train = QCheckBox(self)
        self.train.setGeometry(QRect(730, 646, 90, 20))
        self.train.setCheckable(True)
        self.train.setStyleSheet("QCheckBox{\n"
                                 "    background:transparent;\n"
                                 "    color:black;\n"
                                 "    box-shadow: 1px 1px 1px rgba(0,0,0,0.3);font-size:16px;font-family: 微软雅黑;\n"
                                 "}\n"
                                 "QCheckBox::indicator{\n"
                                 "    width:16px;\n"
                                 "    height:16px;\n"
                                 "}\n"
                                 "QCheckBox::indicator:unchecked{\n"
                                 "    background:#EEE8CD;\n"
                                 "}\n"
                                 "QCheckBox::indicator:checked{\n"
                                 "    background:#BDD8F2;\n"
                                 "}\n"
                                 )
        self.train.setObjectName("train")
        self.train.stateChanged.connect(self.switch_mode)
        self.train.setText(QCoreApplication.translate("MainWindow", "Train"))

        self.test = QCheckBox(self)
        self.test.setGeometry(QRect(805, 646, 90, 20))
        self.test.setCheckable(True)
        self.test.setStyleSheet("QCheckBox{\n"
                                "    background:transparent;\n"
                                "    color:black;\n"
                                "    box-shadow: 1px 1px 1px rgba(0,0,0,0.3);font-size:16px;font-family: 微软雅黑;\n"
                                "}\n"
                                "QCheckBox::indicator{\n"
                                "    width:16px;\n"
                                "    height:16px;\n"
                                "}\n"
                                "QCheckBox::indicator:unchecked{\n"
                                "    background:#EEE8CD;\n"
                                "}\n"
                                "QCheckBox::indicator:checked{\n"
                                "    background:#BDD8F2;\n"
                                "}\n"
                                )
        self.test.setObjectName("test")
        self.test.stateChanged.connect(self.switch_mode)
        self.test.setText(QCoreApplication.translate("MainWindow", "Test"))

        self.serverPort = QLineEdit(self)
        self.serverPort.setGeometry(QRect(718, 690, 150, 30))
        self.serverPort.setStyleSheet("QLineEdit{\n"
                                      "    background:#E6E6E6;\n"
                                      "    color:black;\n"
                                      "    box-shadow: 1px 1px 3px rgba(0,0,0,0.3);font-size:16px;border-radius:5px;font-family: 微软雅黑;\n"
                                      "}\n"
                                      "QLineEdit:hover{                    \n"
                                      "    background:#B3B3B3;\n"
                                      "}\n")
        self.serverPort.setObjectName("serverPort")
        self.serverPort.setText(QCoreApplication.translate("MainWindow", "40952"))
        self.serverPort.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        self.ipAddress = QLineEdit(self)
        self.ipAddress.setGeometry(QRect(718, 740, 150, 30))
        self.ipAddress.setStyleSheet("QLineEdit{\n"
                                     "    background:#E6E6E6;\n"
                                     "    color:black;\n"
                                     "    box-shadow: 1px 1px 3px rgba(0,0,0,0.3);font-size:16px;border-radius:5px;font-family: 微软雅黑;\n"
                                     "}\n"
                                     "QLineEdit:hover{                    \n"
                                     "    background:#B3B3B3;\n"
                                     "}\n")
        self.ipAddress.setObjectName("ipAddress")
        self.ipAddress.setText(QCoreApplication.translate("MainWindow", "10.170.12.5"))
        self.ipAddress.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        self.startButton = QPushButton(self)
        self.startButton.setGeometry(QRect(676, 800, 100, 50))
        self.startButton.setStyleSheet("QPushButton{\n"
                                       "    background:#55AA66;\n"
                                       "    color:white;\n"
                                       "    box-shadow: 1px 1px 3px rgba(0,0,0,0.3);font-size:18px;border-radius:18px;font-family: 微软雅黑;\n"
                                       "}\n"
                                       "QPushButton:hover{                    \n"
                                       "    background:#44BB5C;\n"
                                       "}\n"
                                       "QPushButton:pressed{\n"
                                       "    border: 1px solid #3C3C3C!important;\n"
                                       "    background:#3CC457;\n"
                                       "}")
        self.startButton.setObjectName("startButton")
        self.startButton.setText(QCoreApplication.translate("MainWindow", "START"))
        self.startButton.clicked.connect(self.start_on_click)

    def min_on_click(self):
        self.setWindowState(Qt.WindowMinimized)

    def exit_on_click(self):
        self.clear_all()
        # QCoreApplication.instance().quit()
        sys.exit()

    def start_on_click(self):

        while True:
            preds=self.modeltest()
            if preds == 2:
                self.med.setVisible(True)
            elif preds == 1:
                self.low.setVisible(True)
            else:
                self.high.setVisible(True)
            QApplication.processEvents()  # 刷新界面
        # self.clear_all()


    def modeltest(self):
        model = ACNN(nb_classes=3, Chans=64, Samples=128,
                     dropoutRate=0.5)
        filepath = './checkpoint.h5'
        # load optimal weights
        model.load_weights(filepath=filepath)

        # load test data
        test_data = self.loaddata_online()

        probs = model.predict(test_data)
        preds = probs.argmax(axis=-1)
        return preds


    def loaddata_online(self):
        host = '170.0.0.1'
        sfreq = 1024
        port = 8888
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
        # data = np.reshape(data, (data.shape[0], 1, data.shape[1], data.shape[2]))
        data = np.reshape(data, (1, data.shape[0], data.shape[1], data.shape[2]))
        # data_loader = DataLoader(data, batch_size=1)
        return data

    # 清空所有信息
    def clear_all(self):
        # 清空视频选中
        for i in range(20):
            w = self.grid.itemAt(i).widget()
            w.setChecked(False)
        # 清空模式选中
        self.train.setChecked(False)
        self.test.setChecked(False)
        # 清空 videos列表
        with open('videos.txt', 'w') as f:
            f.truncate()
            f.close()
        # 清空 mask列表
        with open('mask.txt', 'w') as f:
            f.truncate()
            f.close()

    def mouseMoveEvent(self, e: QMouseEvent):
        if e.y() > 200:
            return
        self._endPos = e.pos() - self._startPos
        self.move(self.pos() + self._endPos)

    def mousePressEvent(self, e: QMouseEvent):
        if e.y() > 200:
            return
        if e.button() == Qt.LeftButton:
            self._isTracking = True
            self._startPos = QPoint(e.x(), e.y())

    def mouseReleaseEvent(self, e: QMouseEvent):
        if e.y() > 200:
            return
        if e.button() == Qt.LeftButton:
            self._isTracking = False
            self._startPos = None
            self._endPos = None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
