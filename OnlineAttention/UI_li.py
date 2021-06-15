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
        self.title.setPixmap(QPixmap("./pic/title1.png"))
        self.title.setScaledContents(True)
        self.title.setObjectName("title")

        self.normal = QLabel(self)
        self.normal.setGeometry(QRect(378, 220, 300, 400))
        self.normal.setPixmap(QPixmap("./pic/normal.png"))
        self.normal.setScaledContents(True)
        self.normal.setObjectName("normal")

        self.ease = QLabel(self)
        self.ease.setGeometry(QRect(378, 220, 300, 400))
        self.ease.setPixmap(QPixmap("./pic/ease.png"))
        self.ease.setScaledContents(True)
        self.ease.setObjectName("ease")
        self.ease.setVisible(False)

        self.load = QLabel(self)
        self.load.setGeometry(QRect(378, 220, 300, 400))
        self.load.setPixmap(QPixmap("./pic/load.png"))
        self.load.setScaledContents(True)
        self.load.setObjectName("load")
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
            # res = get_load()
            res = 1
            if res == 1:
                self.ease.setVisible(False)
                self.load.setVisible(True)
            else:
                self.ease.setVisible(True)
                self.load.setVisible(False)
            QApplication.processEvents()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    load = Load()
    load.show()
    sys.exit(app.exec_())

