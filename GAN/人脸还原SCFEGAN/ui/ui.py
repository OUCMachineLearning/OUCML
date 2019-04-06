from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1200, 660)
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(10, 10, 97, 27))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(130, 10, 97, 27))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(250, 10, 97, 27))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(Form)
        self.pushButton_4.setGeometry(QtCore.QRect(370, 10, 97, 27))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(Form)
        self.pushButton_5.setGeometry(QtCore.QRect(560, 360, 81, 27))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(Form)
        self.pushButton_6.setGeometry(QtCore.QRect(490, 40, 97, 27))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_7 = QtWidgets.QPushButton(Form)
        self.pushButton_7.setGeometry(QtCore.QRect(490, 10, 97, 27))
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_8 = QtWidgets.QPushButton(Form)
        self.pushButton_8.setGeometry(QtCore.QRect(370, 40, 97, 27))
        self.pushButton_8.setObjectName("pushButton_8")
        self.graphicsView = QtWidgets.QGraphicsView(Form)
        self.graphicsView.setGeometry(QtCore.QRect(20, 120, 512, 512))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_2 = QtWidgets.QGraphicsView(Form)
        self.graphicsView_2.setGeometry(QtCore.QRect(660, 120, 512, 512))
        self.graphicsView_2.setObjectName("graphicsView_2")

        self.saveImg = QtWidgets.QPushButton(Form)
        self.saveImg.setGeometry(QtCore.QRect(610, 10, 97, 27))
        self.saveImg.setObjectName("saveImg")

        self.arrangement = QtWidgets.QPushButton(Form)
        self.arrangement.setGeometry(QtCore.QRect(610, 40, 97, 27))
        self.arrangement.setObjectName("arrangement")

        self.retranslateUi(Form)
        self.pushButton.clicked.connect(Form.open)
        self.pushButton_2.clicked.connect(Form.mask_mode)
        self.pushButton_3.clicked.connect(Form.sketch_mode)
        self.pushButton_4.clicked.connect(Form.stroke_mode)
        self.pushButton_5.clicked.connect(Form.complete)
        self.pushButton_6.clicked.connect(Form.undo)
        self.pushButton_7.clicked.connect(Form.color_change_mode)
        self.pushButton_8.clicked.connect(Form.clear)

        self.saveImg.clicked.connect(Form.save_img)

        self.arrangement.clicked.connect(Form.arrange)

        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "SC-FEGAN"))
        self.pushButton.setText(_translate("Form", "Open Image"))
        self.pushButton_2.setText(_translate("Form", "Mask"))
        self.pushButton_3.setText(_translate("Form", "Sketches"))
        self.pushButton_4.setText(_translate("Form", "Color"))
        self.pushButton_5.setText(_translate("Form", "Complete"))
        self.pushButton_6.setText(_translate("Form", "Undo"))
        self.pushButton_7.setText(_translate("Form", "Palette"))
        self.pushButton_8.setText(_translate("Form", "Clear"))

        self.saveImg.setText(_translate("Form", "Save Img"))

        self.arrangement.setText(_translate("Form", "Arrange"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
