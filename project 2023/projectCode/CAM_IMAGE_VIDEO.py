from PyQt5.QtWidgets import QLabel, QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
# --------------------
import os
import sys
import numpy as np
import cv2
import imutils
import time
global isvideorun
isvideorun = False
global camera
global exitflag
exitflag = False
global isCamrun
isCamrun = False
global exitflagCam
exitflagCam = False
global net
class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('Safe_unsafe.ui', self)
        self.CameraButton.clicked.connect(self.startCam)
        self.ImageButton.clicked.connect(self.loadImage)
        self.VideoButton.clicked.connect(self.loadVideo)
        self.Label1.setAlignment(QtCore.Qt.AlignCenter)
        self.Label2.setAlignment(QtCore.Qt.AlignCenter)
        self.Label1.setText("Select Option")
        self.Label2.setText("")
        self.show()
        yolo = "coco_file\\"
        self.Setup(yolo)

    def Setup(self, yolo):
        global net, ln, LABELS
        weights = os.path.sep.join([yolo, "yolov3.weights"])
        config = os.path.sep.join([yolo, "yolov3.cfg"])
        labelsPath = os.path.sep.join([yolo, "coco.names"])
        LABELS = open(labelsPath).read().strip().split("\n")
        net = cv2.dnn.readNetFromDarknet(config, weights)
        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def loadImage(self):
        self.Label1.setText(" ")
        filename, _ = QFileDialog.getOpenFileName(self)
        print('filename ', filename)
        self.image = cv2.imread(filename)
        self.setPhoto(self.image)
        self.update()
        cv2.waitKey(1)
        self.ImageProcess(self.image)

        # self.predict(image_np)

    def ImageProcess(self, frame):
        global processedImg
        global net
        (H, W) = (None, None)
        # frame = image.copy()
        if W is None or H is None:
            (H, W) = frame.shape[:2]
# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        starttime = time.time()
        layerOutputs = net.forward(ln)
        stoptime = time.time()
        print("Video is Getting Processed at {:.4f} seconds per frame".format(
            (stoptime-starttime)))
        confidences = []
        outline = []
        for output in layerOutputs:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            for detection in output:
                scores = detection[5:]
                maxi_class = np.argmax(scores)
                confidence = scores[maxi_class]
                if LABELS[maxi_class] == "person":
                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > 0.5:
                        # scale the bounding box coordinates back relative to the
                        # size of the image, keeping in mind that YOLO actually
                        # returns the center (x, y)-coordinates of the bounding
                        # box followed by the boxes' width and height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        # use the center (x, y)-coordinates to derive the top and
                        # and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        # update our list of bounding box coordinates, confidences,
                        # and class IDs
                        outline.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                else:
                    self.Label1.setText(" ")

    # apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
        box_line = cv2.dnn.NMSBoxes(outline, confidences, 0.5, 0.3)

# ensure at least one detection exists
        if len(box_line) > 0:
            flat_box = box_line.flatten()
            for i in flat_box:
                # extract the bounding box coordinates
                (x, y) = (outline[i][0], outline[i][1])
                (w, h) = (outline[i][2], outline[i][3])
                # draw a bounding box rectangle and label on the image
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        NoOfPerson = len(flat_box)
        print("Person Count : " + str(NoOfPerson))
        ThresholdCount=int(self.count.text())
        print("Threshold count : ",ThresholdCount )
        self.Label1.setText("Person Count : " + str(NoOfPerson))
        if NoOfPerson > ThresholdCount:
            self.Label2.setStyleSheet("background-color: red")
            self.Label2.setText(" Unsafe ")
        else:
            self.Label2.setStyleSheet("background-color: green")
            self.Label2.setText("  Safe ")

        self.image = frame
        self.setPhoto(self.image)

    def setPhoto(self, image):
        self.tmp = image
        image = imutils.resize(image, width=640)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(
            frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.ImageToShow.setPixmap(QtGui.QPixmap.fromImage(image))

    def loadVideo(self):
        global isvideorun
        global cam
        global exitflag

        if isvideorun:
            exitflag = True
            cam.release

            self.VideoButton.setText('Video')
            isvideorun = False
            exitflag = True
            print('stopping Video....')
        else:
            print('Starting Video....')
            isvideorun = True
            exitflag = False
            self.VideoButton.setText('Stop')
            filename, _ = QFileDialog.getOpenFileName(self)
            print('filename ', filename)
            cam = cv2.VideoCapture(filename)
            self.Label1.setText(" ")

            # image_np = cv2.imread(filename)
            # self.predict(image_np)
            while exitflag == False:
                # if exitflag:
                #     break
                ret, image_np = cam.read()
                # self.predict(image_np)
                self.ImageProcess(image_np)
                self.update()
                cv2.waitKey(1)

    def startCam(self):
        global isCamrun
        global camera
        global exitflagCam
        if isCamrun:
            exitflagCam = True
            camera.release
            self.CameraButton.setText('Cam')
            isCamrun = False
            exitflagCam = True
            print('Stopping Camera.....')
        else:
            print('Starting Camera.....')
            isCamrun = True
            exitflagCam = False
            self.CameraButton.setText('Stop')

            camera = cv2.VideoCapture(1)
            cv2.waitKey(1000)

            while exitflagCam == False:
                # if exitflag:
                #     break
                try:
                    ret, image_np = camera.read()
                    image_np = cv2.flip(image_np, 1)
                    # self.predict(image_np)
                    self.ImageProcess(image_np)
                    self.update()
                    cv2.waitKey(1)
                except:
                    print('frame cant read from camera')
                    pass

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
