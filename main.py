from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from mask.mainwindow import Ui_MainWindow
import sys
from mask import camera_infer
from facenet_retinaface import predict
class MainWindowShow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindowShow, self).__init__()
        self.setupUi(self)
        self.initUI()
        self.CallBackFunctions()

    def initUI(self):
        self.setWindowIcon(QIcon('mask\icon.icns'))  # 给程序设置图标
        op = QtWidgets.QGraphicsOpacityEffect()  # # 设置透明度的值，0.0到1.0，最小值0是透明，1是不透明
        op.setOpacity(0.5)
        self.setWindowTitle("基于深度学习的口罩检测及人脸识别系统")
        self.DispLb = QtWidgets.QLabel(self.frame)
        self.DispLb.setPixmap(QtGui.QPixmap("mask\h.png"))
        self.DispLb.setObjectName("DispLb")

    def discernment(self):
        QMessageBox.information(self, '提示', '如果有任何问题请联系我,个人邮箱:hhanqikai@163.com', QMessageBox.Yes)

    def realtimetest(self):
        args = camera_infer.parse_args()
        camera_infer.predict_video(args)

    def facedetect(self):
        predict.detect()

    def CallBackFunctions(self):  # 回调函数
        self.pushButton_5.clicked.connect(self.discernment)
        self.pushButton_4.clicked.connect(self.close)
        self.pushButton_3.clicked.connect(self.realtimetest)
        self.pushButton_8.clicked.connect(self.facedetect)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('icon.icns'))
    ui = MainWindowShow()
    ui.show()
    sys.exit(app.exec_())
