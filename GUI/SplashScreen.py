from PyQt5 import uic, QtCore
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMainWindow, QGraphicsDropShadowEffect
from MainScreen import Window
import queue
import time

from TestModelsWorkerThread import TestModelsThread



class SplashScreen(QMainWindow):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.main = None
        self.ui = uic.loadUi(r".\UI\SplashScreen.ui", self)

        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        ## DROP SHADOW EFFECT
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0, 0, 0, 60))
        self.ui.dropShadowFrame.setGraphicsEffect(self.shadow)

        ## QTIMER ==> START
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.progress)
        # TIMER IN MILLISECONDS
        self.timer.start(20)

        # create threads
        self.label_loading.setText('Creating background threads...')

        self.q = queue.Queue()
        self.worker = TestModelsThread(self.q)
        self.worker.results.connect(self.worker_event)
        self.worker.daemon = True
        self.worker.start()
        self.counter += 10
        # SET VALUE TO PROGRESS BAR
        self.ui.progressBar.setValue(self.counter)
        # load deepfill model
        self.q.put([2, r'E:\FinalProject\TrainedModels\release_places2_256_deepfill_v2'])
        # load yolo model
        self.q.put([1, r'E:\FinalProject\TrainedModels\YOLOv3\yolov3'])

        self.show()

    def worker_event(self, event_data):
        if event_data[0] == 1:
            self.counter += 50
            if event_data[1]:
                self.label_loading.setText('YOLOv3 loaded successfully')
            else:
                self.label_loading.setText('Failed to load YOLOv3')

        if event_data[0] == 2:
            self.counter += 40
            if event_data[1]:
                self.label_loading.setText('DeepFillV1 loaded successfully')
            else:
                self.label_loading.setText('Failed to load DeepFillV1')

        if event_data[0] == 3:
            self.label_loading.setText(event_data[1])

    def progress(self):
        # SET VALUE TO PROGRESS BAR
        self.ui.progressBar.setValue(self.counter)

        # CLOSE SPLASH SCREE AND OPEN APP
        if self.counter >= 100:
            # STOP TIMER
            self.timer.stop()

            # SHOW MAIN WINDOW
            self.main = Window(self.worker, self.q)
            self.main.show()

            # CLOSE SPLASH SCREEN
            self.close()
