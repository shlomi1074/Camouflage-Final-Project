import sys
import os
import shutil
from multiprocessing import Process
import subprocess
from PyQt5 import uic, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow
from ButtonCommands import GuiFunctions


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        # LOAD UI FILE
        self.ui = uic.loadUi(r".\UI\MainScreen.ui", self)
        with open('UI/tab.css', "r") as fh:
             tw = fh.read()
        with open('UI/push_button.css', "r") as fh:
             pb = fh.read()

        self.tabWidget.setStyleSheet(tw)
        self.pushButton_5.setStyleSheet(pb)
        self.yoloStartTrainButton.setStyleSheet(pb)
        self.pushButton_3.setStyleSheet(pb)
        self.yoloStopTrainButton.setStyleSheet(pb)
        self.commands = GuiFunctions(416, 0.5)
        self.yoloStartTrainButton.clicked.connect(self.start_yolo_train)
        self.yoloStopTrainButton.clicked.connect(self.stop_yolo_train)
        self.thread = None
        self.process = None
        self.tensorboard_process = None

    def start_yolo_train(self):
        self.process = Process(target=self.commands.on_yolov3_train_button_click,
                               args=(r'E:\FinalProject\Code\Application\data\logs\yolo_logs',
                                     r'E:\FinalProject\temp\yolo_model',))
        self.process.start()
        self.run_tensorboard('E:\FinalProject\Code\Application\data\logs\yolo_logs')
        self.yoloStartTrainButton.setEnabled(False)

    def stop_yolo_train(self):
        if self.process is not None:
            self.process.terminate()
            self.process = None
        if self.tensorboard_process is not None:
            self.tensorboard_process.kill()
            self.tensorboard_process = None
        self.yoloStartTrainButton.setEnabled(True)

    def run_tensorboard(self, log_dir):
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        self.tensorboard_process = subprocess.Popen(['tensorboard', '--logdir', log_dir])


if __name__ == '__main__':
    app = QApplication([])
    window = Window()
    window.show()
    sys.exit(app.exec_())
