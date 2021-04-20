# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'xiaoxiaoleUI.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(691, 860)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(280, 780, 131, 81))
        self.pushButton.setStyleSheet("QPushButton{\n"
"                color: rgb(255, 255, 255);\n"
"                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgb(166,164,208), stop:0.3 rgb(171,152,230), stop:1 rgb(152,140,220));\n"
"                border:1px;\n"
"                border-radius:5px; /*border-radius控制圆角大小*/\n"
"                padding:2px 4px;  \n"
"            }\n"
"            QPushButton:hover{\n"
"                color: rgb(255, 255, 255); \n"
"                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgb(130,120,226), stop:0.3 rgb(120,130,230), stop:1 rgb(125,140,226));\n"
"                border:1px;  \n"
"                border-radius:5px; /*border-radius控制圆角大小*/\n"
"                padding:2px 4px; \n"
"            }\n"
"            QPushButton:pressed{    \n"
"                color: rgb(255, 255, 255); \n"
"                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgb(240,156,121), stop:0.3 rgb(220,160,140), stop:1 rgb(230,140,120));  \n"
"                border:1px;  \n"
"                border-radius:5px; /*border-radius控制圆角大小*/\n"
"                padding:2px 4px; \n"
"            }")
        self.pushButton.setFlat(False)
        self.pushButton.setObjectName("pushButton")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 691, 860))
        self.label.setStyleSheet("/*QLabel{background-color: rgb(255, 170, 255);\n"
"background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 rgba(170, 170, 255), stop:1 rgba(255, 170, 255, 255));}\n"
"\n"
"")
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(170, 0, 361, 521))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label.raise_()
        self.label_2.raise_()
        self.pushButton.raise_()
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "XiaoXiaoLe"))
        self.pushButton.setText(_translate("MainWindow", "开始演示"))
