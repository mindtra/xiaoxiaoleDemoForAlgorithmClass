# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'xiaoxiaoleUI.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!
import os
import sys
from PIL.ImageQt import ImageQt
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QApplication

from xiaoXiaoLe.xiaoxiaoleUI_form import Ui_MainWindow


class MainWindows(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)

        self.setupUi(self)
        self.timer = QtCore.QTimer(self)
        self.pushButton.clicked.connect(self.on_clicked)
        self.path = r"E:\算法设计与分析实验\实验3\example"
        image_path = self.path + "\\background.jpg"

        # 清晰度处理
        im = Image.open(image_path)
        im = im.resize((self.label.width(), self.label.height()), Image.ANTIALIAS)
        im.save(self.path+"\\background_1.jpeg", "JPEG", quality=95, subsampling=0)
        print('ok')
        # qim = ImageQt(im)
        image_path = self.path + "\\background_1.jpeg"

        background = QtGui.QPixmap(image_path).scaled(self.label.width(), self.label.height(),
                                                      aspectRatioMode=Qt.KeepAspectRatio)

        # background = QtGui.QPixmap.fromImage(qim).scaled(self.label.width(), self.label.height(),
        #                                                  aspectRatioMode=Qt.KeepAspectRatio)
        self.label.setPixmap(background)
        self.imageNum = 0

    def on_clicked(self):
        self.pushButton.setEnabled(False)
        print('单击了OK按钮')
        # self.scatterPlot()
        self.pushButton.setText("演示中···")
        # time.sleep(self.t_num)
        self.label_2.setPixmap(QtGui.QPixmap(""))
        self.timer_start()
        print('plot finished')

    def timer_start(self):
        self.timer.timeout.connect(self.picturePlot)
        self.timer.start(1500)

    def picturePlot(self):
        x = (self.label.x() + self.label.width()) // 2

        image_path = self.path + "\\example_" + str(self.imageNum) + '.png'
        if os.path.isfile(image_path):
            image = QtGui.QPixmap(image_path)

            self.label_2.setGeometry(QtCore.QRect(x - image.width() // 2, 10, image.width(), image.height()))
            self.label_2.setPixmap(image)
            print("success in %d" % self.imageNum)
            self.imageNum += 1
        else:
            print("image read failed.")
            self.timer.stop()
            self.timer.deleteLater()
            self.timer = QtCore.QTimer(self)
            self.imageNum = 0
            self.pushButton.setText("重新演示？")
            print("Start again?")
            self.pushButton.setEnabled(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindows()
    win.show()
    app.exec_()
