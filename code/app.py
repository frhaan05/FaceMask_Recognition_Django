from pkg_resources import SOURCE_DIST
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QInputDialog, QProgressBar, QDialog

import os, sys
import cv2 ### pip install opencv-python
import numpy as np ## pip install numpy
import json

# import keras for loading the classifier
from tensorflow.keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from networks import Facenet
from utils.utils import compare_faces, preprocess_face
from utils.face_alignment import FaceAligner
from face_recognition import add_to_dataset, delete_from_dataset
from mtcnn import MTCNN

import CONFIG


class Detect(QThread):
    changePixmap = pyqtSignal(QImage)
    def __init__(self, source, face_detector, face_aligner, facenet, mobilenet):
        super().__init__()
        self.cap = None
        self.top_n = 3

        self.face_detector = face_detector
        self.face_aligner = face_aligner
        self.facenet = facenet
        self.mobilenet = mobilenet
        self.is_FR = True
        self.source = source


        if os.path.exists(CONFIG.FR_PATH_TO_CLASSIFIER):
            # load face recognition classifier
            self.clr = load_model(CONFIG.FR_PATH_TO_CLASSIFIER)
        else: 
            self.clr = None

        if os.path.exists(CONFIG.FR_PATH_TO_CLASSES_FILE):
            # load classes
            with open(CONFIG.FR_PATH_TO_CLASSES_FILE, 'r') as fp:
                self.classes = json.load(fp)
        else:
            self.classes = None
        
        if os.path.exists(CONFIG.FR_PATH_TO_DATASET_FILE):
            # load data
            data = np.load(CONFIG.FR_PATH_TO_DATASET_FILE)
            self.X = data["features"]
            self.y = data["labels"]
        else:            
            self.X = None
            self.y = None
    
    def processFrame(self, frame_orig):
        frame = frame_orig.copy()
        # convert the frame into RGB image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # detect faces using MTCNN face detector
        faces = self.face_detector.detect_faces(frame)

        if len(faces) == 0:
            return

        # iterate through every face
        for face in faces:

            landmarks, box, score = face["keypoints"], face["box"], face["confidence"]
            # if face confidnece score is less than 60% then skip
            if score < 0.8:
                return
            
            # extract the bounding box information from the face
            x1, y1, width, height = box
            x2, y2 = x1 + width, y1 + height

            face_image = frame[y1:y2, x1:x2]
            face_mobilenetv2 = face_image.copy()
            # resize the face to the target size of the classifier
            face_mobilenetv2 = cv2.resize(face_mobilenetv2, CONFIG.IMAGE_INPUT_SIZE)
            # normalize the face array and convert to float 32
            face_mobilenetv2 = (face_mobilenetv2 / 255.).astype(np.float32)
            # add extra dimension to the image as deep learning models accept images in batch
            face_mobilenetv2 = np.expand_dims(np.array(face_mobilenetv2), axis=0)
            # perform the predictions
            prediction = self.mobilenet.predict(face_mobilenetv2)[0]
            # get the index if the class with highest probability
            index = np.argmax(prediction)
            # get the label and probability
            label = CONFIG.MOBILENET_CLASSES[index]
            prob = round(prediction[index] * 100, 2)

            # depending on the class, draw the bounding box on the frame
            if label == "Correctly Masked Face":
                color = (0, 255, 0)
            elif label == "Incorrectly Masked Face":
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)

            str_ = label + ": " + str(prob) + "%"
            if label in ["Incorrectly Masked Face", "Unmasked Face"] and self.is_FR:
                # first align the face
                face_image = self.face_aligner.align(frame, landmarks['right_eye'], landmarks['left_eye'])

                # do face recognition
                face_facenet = preprocess_face(face_image, CONFIG.FACENET_INPUT_SIZE)
                embeddings = self.facenet.predict(face_facenet)

                # predict the person based on embeddings
                predictions = self.clr.predict(embeddings)[0]
                # predictions with top three probs
                predictions_top_3 = (-predictions).argsort()[:self.top_n]
                # taking the top-3 predictions and then extracting their embeddings
                known_embeddings = np.array([np.mean(self.X[np.where(self.y == _idx)], axis=0) for _idx in predictions_top_3])
                # comparing top-3 predicted person's embeddings with query person
                result = compare_faces(known_embeddings, embeddings[0].tolist(), tolerance=0.47)
                # post-processing final result
                name = self.classes[str(predictions_top_3[result])] if result is not None else "unknown"
                str_ = str_ + ' ' + name

            # Crop the image frame into rectangle
            cv2.rectangle(frame_orig, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_orig, str_, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
        return cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)

    def run(self):

        if self.classes is None or self.X is None or self.y is None or self.clr is None:
            self.is_FR = False

        if type(self.source) == list:
            for image in self.source:
                if os.path.exists(image):
                    print("processing:", image)
                    try:
                        frame = cv2.imread(image)
                        h, w, c = frame.shape
                        step = c * w
                        frame = self.processFrame(frame)
                        qImg1 = QImage(frame.data, w, h, step, QImage.Format_RGB888)
                        self.changePixmap.emit(qImg1)
                    except Exception as e:
                        print(f"ERROR -> {str(e)}")
        else:
            try:
                self.cap = cv2.VideoCapture(self.source)
            except:
                return

            while True:
                try:
                    # Capture frame-by-frame
                    ret, frame_orig = self.cap.read()
                    if ret:
                        h, w, c = frame_orig.shape
                        step = c * w

                        frame_orig = self.processFrame(frame_orig)

                        qImg1 = QImage(frame_orig.data, w, h, step, QImage.Format_RGB888)
                        self.changePixmap.emit(qImg1)
                        # # frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                except Exception as e:
                    print(f"ERROR -> {str(e)}")

class Ui_MainWindow(object):

    def __init__(self):
        super().__init__()
        self.setupUi(MainWindow)

        self.source = None
        self.face_detector = None
        self.face_aligner = None
        self.facenet_model = None
        self.mobilenet_model = None
        self.fr_clf = None
        self.fr_embeddings = None
        self.fr_labels = None
        self.known_people = None

        # load models and other stuff here
        self.LoadModels()

        # intialize saveTimer
        self.saveTimer = QTimer()
        self.saveTimer.stop()

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
        self.btn_run.clicked.connect(self.handleDetectEvent)

        self.btn_run_images = QtWidgets.QPushButton(self.centralwidget)
        self.btn_run_images.setGeometry(QtCore.QRect(10, 120, 201, 70))
        self.btn_run_images.setObjectName("btn_run_images")
        self.btn_run_images.clicked.connect(self.runImages)

        self.btn_register = QtWidgets.QPushButton(self.centralwidget)
        self.btn_register.setGeometry(QtCore.QRect(10, 220, 201, 91))
        self.btn_register.setObjectName("btn_register")
        self.btn_register.clicked.connect(self.registerPerson)

        self.btn_delete = QtWidgets.QPushButton(self.centralwidget)
        self.btn_delete.setGeometry(QtCore.QRect(10, 340, 201, 91))
        self.btn_delete.setObjectName("btn_delete")
        self.btn_delete.clicked.connect(self.deletePerson)

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

    def LoadModels(self):
        print("loading models...")

        if os.path.exists(CONFIG.FR_PATH_TO_CLASSES_FILE):
            # load classes
            with open(CONFIG.FR_PATH_TO_CLASSES_FILE, 'r') as fp:
                self.known_people = json.load(fp)
        else:            
            self.showMessageDialog("Error", "File Not Found", f"{CONFIG.FR_PATH_TO_CLASSES_FILE} cannot be found...", "Error")

        # load face detector model
        self.face_detector = MTCNN()
        # create Face Alignment objects
        self.face_aligner = FaceAligner(desiredFaceWidth=CONFIG.IMAGE_INPUT_SIZE[0])

        if os.path.exists(CONFIG.FR_FACENET_WEIGHTS):
            # load facenet model
            self.facenet_model = Facenet.loadModel(weights=CONFIG.FR_FACENET_WEIGHTS)
        else:            
            self.showMessageDialog("Error", "File Not Found", f"{CONFIG.FR_FACENET_WEIGHTS} cannot be found. Face Recognition won't work !!!", "Error")

        if os.path.exists(CONFIG.MOBILENET_MODEL_PATH):
                    # load trained classifier
            self.mobilenet_model = load_model(CONFIG.MOBILENET_MODEL_PATH)
        else:            
            self.showMessageDialog("Error", "File Not Found", f"{CONFIG.MOBILENET_MODEL_PATH} cannot be found. Mask Detection won't work !!!", "Error")
        print("models loaded successfully...")

    # @pyqtSlot()
    def setImage(self, qImg1):
        self.label_preview.setPixmap(QPixmap.fromImage(qImg1))

    def registerPerson(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(MainWindow, "QFileDialog.getOpenFileNames()", "","Images (*.jpg, *.jpeg, *.JPG)", options=options)
        if files:
            name, ret = QInputDialog.getText(MainWindow, 'Input Dialog', 'Enter the name of the Person to Register:')
            if ret:
                print("registering person...")
                classes, status = add_to_dataset(name, files)
                if status:
                    self.showMessageDialog("Info", "Person Registered Successfully...", "", "Information")
                    self.known_people = classes
                else:
                    self.showMessageDialog("Error", "Person Already Exists", "Please enter another name...", "Error")
                
    def deletePerson(self):
        name, ret = QInputDialog.getText(MainWindow, 'Input Dialog', 'Enter the name of the Person to delete from database:')
        if ret:
            classes, status = delete_from_dataset(name)
            if status:
                self.showMessageDialog("Info", "Person Deleted Successfully...", "", "Information")
                self.known_people = classes
            else:
                self.showMessageDialog("Error", "Person Couldn't be Found", "Please enter another name...", "Error")

    def showRegisteredPeople(self):
        if self.known_people is not None:
            self.text_records.clear()
            for key, value in self.known_people.items():
                self.text_records.append(value)

    def handleDetectEvent(self):
        if not self.saveTimer.isActive():
            # intialize webcam
            source = 0
            self.saveTimer.start()
            self.detect = Detect(
                source,
                self.face_detector,
                self.face_aligner,
                self.facenet_model,
                self.mobilenet_model)

            self.detect.changePixmap.connect(self.setImage)
            self.detect.start()
            self.btn_run.setText("Stop Detection")
        else:
            # stop writing
            self.detect.cap.release()
            self.saveTimer.stop()                    
            self.detect.terminate()
            self.btn_run.setText("Start Detection")

    def stopDetection(self):
        if self.saveTimer.isActive():
            self.detect.cap.release()
            self.saveTimer.stop()                    
            self.detect.terminate()
            self.btn_run.setText("Start Detection")

    def runImages(self):
        self.stopDetection()
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(MainWindow, "QFileDialog.getOpenFileNames()", "","Images (*.jpg, *.jpeg, *.JPG)", options=options)
        if files:
            print("images loaded:", files)
            # intialize webcam
            self.saveTimer.start()
            self.detect = Detect(
                files,
                self.face_detector,
                self.face_aligner,
                self.facenet_model,
                self.mobilenet_model)

            self.detect.changePixmap.connect(self.setImage)
            self.detect.start()


    def onCloseApplication(self):
        print("closing PyQtTest")
        self.detect.cap.release()
        self.saveTimer.stop()                    
        self.detect.terminate()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    MainWindow.show()

    ret = app.exec_()

    # before closing application, terminate the active processes
    ui.onCloseApplication()
    sys.exit(ret)
