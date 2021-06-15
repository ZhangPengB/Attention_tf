import sys
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QRect
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QMouseEvent
# from analysis import analysis, get_load


import numpy as np

from OnlineAttention.get_data import ActiveTwo

from OnlineAttention.ACNN import ACNN
from braindecode.datautil.signalproc import highpass_cnt, bandpass_cnt, exponential_running_demean, \
    exponential_running_standardize

class Load(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.count = 0

    def initUI(self):

        self.setObjectName('MainWindow')
        self.resize(1024, 768)
        self.setWindowFlags(Qt.FramelessWindowHint)  # 去边框
        self.setAttribute(Qt.WA_TranslucentBackground)  # 设置窗口背景透明

        self.background = QLabel(self)
        self.background.setGeometry(QRect(0, 0, 1024, 768))
        self.background.setPixmap(QPixmap("./pic/background.png"))
        self.background.setScaledContents(True)
        self.background.setObjectName("background")

        self.title = QLabel(self)
        self.title.setGeometry(QRect(318, 100, 400, 80))
        self.title.setPixmap(QPixmap("./pic/title.png"))
        self.title.setScaledContents(True)
        self.title.setObjectName("title")


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


        #
        # self.normal = QLabel(self)
        # self.normal.setGeometry(QRect(378, 220, 300, 400))
        # self.normal.setPixmap(QPixmap("./pic/normal.png"))
        # self.normal.setScaledContents(True)
        # self.normal.setObjectName("normal")
        #
        # self.ease = QLabel(self)
        # self.ease.setGeometry(QRect(378, 220, 300, 400))
        # self.ease.setPixmap(QPixmap("./pic/ease.png"))
        # self.ease.setScaledContents(True)
        # self.ease.setObjectName("ease")
        # self.ease.setVisible(False)

        # self.load = QLabel(self)
        # self.load.setGeometry(QRect(378, 220, 300, 400))
        # self.load.setPixmap(QPixmap("./pic/load.png"))
        # self.load.setScaledContents(True)
        # self.load.setObjectName("load")
        self.load.setVisible(False)

        self.minButton = QPushButton(self)
        self.minButton.setGeometry(QRect(890, 80, 16, 16))
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
        self.exitButton.setGeometry(QRect(920, 80, 16, 16))
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



        self.startButton = QPushButton(self)
        self.startButton.setGeometry(QRect(468, 640, 100, 45))
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
        # QCoreApplication.instance().quit()
        sys.exit()

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


    # 主控制程序
    # def start_on_click(self):  # 本地
    #     i = 145
    #     while True:
    #         val = analysis(i)
    #         if val == 1:
    #             self.ease.setVisible(False)
    #             self.load.setVisible(True)
    #         else:
    #             self.ease.setVisible(True)
    #             self.load.setVisible(False)
    #         i += 1
    #         QApplication.processEvents()

    def start_on_click(self):  # 离线
        i = 1
        while True:
            print(i)
            i += 1
            preds = self.modeltest()

            if preds == 2:
                self.med.setVisible(True)
                self.high.setVisible(False)
                self.low.setVisible(False)
            elif preds == 1:
                self.med.setVisible(False)
                self.high.setVisible(False)
                self.low.setVisible(True)
            else:
                self.med.setVisible(False)
                self.low.setVisible(False)
                self.high.setVisible(True)
            QApplication.processEvents()



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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    load = Load()
    load.show()
    sys.exit(app.exec_())

