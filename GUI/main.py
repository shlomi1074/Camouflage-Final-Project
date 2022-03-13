import sys
import os
import shutil
from multiprocessing import Process
import subprocess
from PyQt5 import uic, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QFileDialog
from ButtonCommands import GuiFunctions


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.yolov3_output_folder = "C:\\trainedModels\\yolov3"
        self.yolov3_tensorboard_logs_folder = "C:\\tensorboard\\logs\\yolov3"
        self.deepfillv1_output_folder = "C:\\trainedModels\\deepfillv1"
        self.deepfillv1_tensorboard_logs_folder = "C:\\tensorboard\\logs\\deepfillv1"
        # LOAD UI FILE
        self.ui = uic.loadUi(r".\UI\MainScreen.ui", self)
        with open('UI/tab.css', "r") as fh:
             tw = fh.read()
        with open('UI/push_button.css', "r") as fh:
             pb = fh.read()

        self.resize(1200, 800)
        self.tabWidget.setStyleSheet(tw)
        self.pushButton_5.setStyleSheet(pb)
        self.yoloStartTrainButton.setStyleSheet(pb)
        self.pushButton_3.setStyleSheet(pb)
        self.yoloStopTrainButton.setStyleSheet(pb)
        self.commands = GuiFunctions(416, 0.5)
        self.tabWidget.setCurrentIndex(0)
        self.saveModelButton.setStyleSheet(pb)
        self.tensorboardLogFolderButton.setStyleSheet(pb)
        self.saveModelButton_2.setStyleSheet(pb)
        self.tensorboardLogFolderButton_2.setStyleSheet(pb)
        self.saveModelButton.setText(self.yolov3_output_folder)
        self.saveModelButton_2.setText(self.deepfillv1_output_folder)
        self.tensorboardLogFolderButton.setText(self.yolov3_tensorboard_logs_folder)
        self.tensorboardLogFolderButton_2.setText(self.deepfillv1_tensorboard_logs_folder)
        self.trainingTrackerLabel.setVisible(False)
        self.yoloStartTrainButton.clicked.connect(self.start_yolo_train)
        self.yoloStopTrainButton.clicked.connect(self.stop_yolo_train)
        self.saveModelButton.clicked.connect(self.yolov3_select_model_folder)
        self.tensorboardLogFolderButton.clicked.connect(self.yolov3_select_logs_folder)
        self.saveModelButton.clicked.connect(self.deepfillv1_select_model_folder)
        self.tensorboardLogFolderButton.clicked.connect(self.deepfillv1_select_logs_folder)
        self.process = None
        self.tensorboard_process = None

    def yolov3_select_model_folder(self):
        dir = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if dir != '' and dir is not None:
            self.yolov3_output_folder = dir
        self.saveModelButton.setText(self.yolov3_output_folder)

    def yolov3_select_logs_folder(self):
        dir = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if dir != '' and dir is not None:
            self.yolov3_tensorboard_logs_folder = dir
        self.tensorboardLogFolderButton.setText(self.yolov3_tensorboard_logs_folder)

    def deepfillv1_select_model_folder(self):
        dir = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if dir != '' and dir is not None:
            self.deepfillv1_output_folder = dir
        self.saveModelButton.setText(self.deepfillv1_output_folder)

    def deepfillv1_select_logs_folder(self):
        dir = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if dir != '' and dir is not None:
            self.deepfillv1_tensorboard_logs_folder = dir
        self.tensorboardLogFolderButton.setText(self.deepfillv1_tensorboard_logs_folder)

    def start_yolo_train(self):
        self.process = Process(target=self.commands.on_yolov3_train_button_click,
                               args=(r'..\Code\Application\data\logs\yolo_logs',
                                     r'..\temp\yolo_model',))
        self.process.start()
        self.run_tensorboard(r'C:\Camouflage-Final-Project\Code\Application\data\logs\yolo_logs')
        self.trainingTrackerLabel.setVisible(True)
        self.yoloStartTrainButton.setEnabled(False)

    def stop_yolo_train(self):
        if self.process is not None:
            self.process.terminate()
            self.process = None
        if self.tensorboard_process is not None:
            self.tensorboard_process.kill()
            self.tensorboard_process = None
        self.yoloStartTrainButton.setEnabled(True)
        self.trainingTrackerLabel.setVisible(False)

    def run_tensorboard(self, log_dir):
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        self.tensorboard_process = subprocess.Popen(['tensorboard', '--logdir', log_dir])

        # self.tensorboard_process.wait()
        # print(self.tensorboard_process.poll())


if __name__ == '__main__':
    app = QApplication([])
    window = Window()
    window.show()
    sys.exit(app.exec_())