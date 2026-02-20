from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QInputDialog, QProgressBar, QDialog

import os, sys


class Ui_MainWindow(object):
    def __init__(self):
        super().__init__()
        self.setupUi(MainWindow)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1080, 720)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.lbl_preview_text = QtWidgets.QLabel(self.centralwidget)
        self.lbl_preview_text.setGeometry(QtCore.QRect(230, 0, 581, 31))
        font = QtGui.QFont()
        font.setPointSize(16)

        self.lbl_preview_text.setFont(font)
        self.lbl_preview_text.setObjectName("lbl_preview_text")

        self.label_preview = QtWidgets.QLabel(self.centralwidget)
        self.label_preview.setGeometry(QtCore.QRect(240, 60, 591, 591))
        self.label_preview.setScaledContents(True)
        self.label_preview.setObjectName("label_preview")

        self.btn_run = QtWidgets.QPushButton(self.centralwidget)
        self.btn_run.setGeometry(QtCore.QRect(10, 20, 201, 70))
        self.btn_run.setObjectName("btn_run")
        # self.btn_run.clicked.connect(self.handleDetectEvent)

        self.btn_run_images = QtWidgets.QPushButton(self.centralwidget)
        self.btn_run_images.setGeometry(QtCore.QRect(10, 120, 201, 70))
        self.btn_run_images.setObjectName("btn_run_images")
        self.btn_run_images.clicked.connect(self.runImages)

        self.btn_register = QtWidgets.QPushButton(self.centralwidget)
        self.btn_register.setGeometry(QtCore.QRect(10, 220, 201, 91))
        self.btn_register.setObjectName("btn_register")
        # self.btn_register.clicked.connect(self.registerPerson)

        self.btn_delete = QtWidgets.QPushButton(self.centralwidget)
        self.btn_delete.setGeometry(QtCore.QRect(10, 340, 201, 91))
        self.btn_delete.setObjectName("btn_delete")
        # self.btn_delete.clicked.connect(self.deletePerson)

        self.btn_show_registered_people = QtWidgets.QPushButton(self.centralwidget)
        self.btn_show_registered_people.setGeometry(QtCore.QRect(10, 450, 201, 91))
        self.btn_show_registered_people.setObjectName("btn_show_registered_people")
        self.btn_show_registered_people.clicked.connect(self.showRegisteredPeople)

        self.btn_settings = QtWidgets.QPushButton(self.centralwidget)
        self.btn_settings.setGeometry(QtCore.QRect(10, 570, 201, 91))
        self.btn_settings.setObjectName("btn_settings")

        self.lbl_preview_text_2 = QtWidgets.QLabel(self.centralwidget)
        self.lbl_preview_text_2.setGeometry(QtCore.QRect(860, 10, 181, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.lbl_preview_text_2.setFont(font)
        self.lbl_preview_text_2.setObjectName("lbl_preview_text_2")

        self.list_record = QtWidgets.QListWidget(self.centralwidget)
        self.list_record.setGeometry(QtCore.QRect(860, 50, 211, 281))
        self.list_record.setObjectName("list_record")

        self.text_records = QtWidgets.QTextBrowser(self.centralwidget)
        self.text_records.setGeometry(QtCore.QRect(860, 421, 211, 251))
        self.text_records.setObjectName("text_records")

        self.lbl_preview_text_3 = QtWidgets.QLabel(self.centralwidget)
        self.lbl_preview_text_3.setGeometry(QtCore.QRect(860, 380, 181, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.lbl_preview_text_3.setFont(font)
        self.lbl_preview_text_3.setObjectName("lbl_preview_text_3")
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1080, 25))
        self.menubar.setObjectName("menubar")

        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionSettings = QtWidgets.QAction(MainWindow)
        self.actionSettings.setObjectName("actionSettings")

        self.menuFile.addAction(self.actionSettings)
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.lbl_preview_text.setText(_translate("MainWindow", "Preview"))
        self.label_preview.setText(_translate("MainWindow", "TextLabel"))
        self.btn_run.setText(_translate("MainWindow", "Run Webcam"))
        self.btn_run_images.setText(_translate("MainWindow", "Run Images"))
        self.btn_register.setText(_translate("MainWindow", "Register Person"))
        self.btn_delete.setText(_translate("MainWindow", "Delete Person"))
        self.btn_show_registered_people.setText(_translate("MainWindow", "Show Registered People"))
        self.btn_settings.setText(_translate("MainWindow", "Advanced Settings"))
        self.lbl_preview_text_2.setText(_translate("MainWindow", "Records"))
        self.list_record.setStatusTip(_translate("MainWindow", "Unmasked People Records"))
        self.lbl_preview_text_3.setText(_translate("MainWindow", "Record Details"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionSettings.setText(_translate("MainWindow", "Settings"))

    def showMessageDialog(self, title, heading, message_str, type):
        msg = QMessageBox()
        if type == "Error":
            msg.setIcon(QMessageBox.Critical)
        elif type == "Information":
            msg.setIcon(QMessageBox.Information)
        msg.setText(heading)
        msg.setInformativeText(message_str)
        msg.setWindowTitle(title)
        msg.exec_()

    def runImages(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(MainWindow, "QFileDialog.getOpenFileNames()", "","Images (*.jpg, *.jpeg, *.JPG)", options=options)
        if files:
            
            name, ret = QInputDialog.getText(MainWindow, 'Input Dialog', 'Enter the name of the Person to Register:')
            if ret:
                self.showMessageDialog("Info", "Person Registered Successfully...", "", "Information")
     
    def showRegisteredPeople(self):
        self.text_records.clear()
        for value in range(20):
            self.text_records.append(str(value))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    MainWindow.show()

    ret = app.exec_()
    sys.exit(ret)
