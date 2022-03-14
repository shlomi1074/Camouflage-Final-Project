import sys
import os
import shutil
from multiprocessing import Process
import subprocess
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from ButtonCommands import GuiFunctions


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        # DEFAULT VALUES
        self.yolov3_output_folder = r'C:\trainedModels\yolov3'
        self.yolov3_tensorboard_logs_folder = r'C:\tensorboard\logs\yolov3'
        self.deepfillv1_output_folder = r'C:\\trainedModels\\deepfillv1'
        self.deepfillv1_tensorboard_logs_folder = r'C:\\tensorboard\\logs\\deepfillv1'

        self.camouflage_api = GuiFunctions(416, 0.5)
        self.action_process = None
        self.tensorboard_process = None

        """
        Init UI
        """
        # LOAD UI FILE
        self.ui = uic.loadUi(r".\UI\MainScreen.ui", self)
        self.setFixedSize(1200, 800)
        # LOAD TAB WIDGET CSS FILE
        with open('UI/tab.css', "r") as fh:
            tw = fh.read()
            self.tabContainer.setStyleSheet(tw)
        self.tabContainer.setCurrentIndex(0)
        self.saveModelLineText.setText(self.yolov3_output_folder)
        self.saveModelLineText_2.setText(self.deepfillv1_output_folder)
        self.tensorboardLogslLineText.setText(self.yolov3_tensorboard_logs_folder)
        self.tensorboardLogslLineText_2.setText(self.deepfillv1_tensorboard_logs_folder)
        self.trainingTrackerLabel.setVisible(False)
        self.tensorboardIcon.setVisible(False)

        """
        click events
        """
        self.yoloStartTrainButton.clicked.connect(self.start_yolo_train)
        self.yoloStopTrainButton.clicked.connect(self.stop_yolo_train)
        self.deepfillStartTrainButton.clicked.connect(self.start_deepfill_train)
        self.deepfillStopTrainButton.clicked.connect(self.stop_deepfill_train)
        self.saveModelButton.mousePressEvent = self.yolov3_select_model_folder
        self.saveModelButton_2.mousePressEvent = self.deepfillv1_select_model_folder
        self.tensorboardLogFolderButton.mousePressEvent = self.yolov3_select_logs_folder
        self.tensorboardLogFolderButton_2.mousePressEvent = self.deepfillv1_select_logs_folder

    def yolov3_select_model_folder(self, event):
        yolo_dir = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if yolo_dir != '' and yolo_dir is not None:
            self.yolov3_output_folder = yolo_dir
        self.saveModelLineText.setText(self.yolov3_output_folder)

    def yolov3_select_logs_folder(self, event):
        tensorboard_yolo_dir = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if tensorboard_yolo_dir != '' and tensorboard_yolo_dir is not None:
            self.yolov3_tensorboard_logs_folder = tensorboard_yolo_dir
        self.tensorboardLogslLineText.setText(self.yolov3_tensorboard_logs_folder)

    def deepfillv1_select_model_folder(self, event):
        deep_dir = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if deep_dir != '' and deep_dir is not None:
            self.deepfillv1_output_folder = deep_dir
        self.saveModelLineText_2.setText(self.deepfillv1_output_folder)

    def deepfillv1_select_logs_folder(self, event):
        tensorboard_deep_dir = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if tensorboard_deep_dir != '' and tensorboard_deep_dir is not None:
            self.deepfillv1_tensorboard_logs_folder = tensorboard_deep_dir
        self.tensorboardLogslLineText_2.setText(self.deepfillv1_tensorboard_logs_folder)

    def start_yolo_train(self):
        if not self.yolo_train_parameter_validation():
            print("Bad parameters")
            return
        self.action_process = Process(target=self.camouflage_api.on_yolov3_train_button_click,
                                      args=(self.yolov3_tensorboard_logs_folder,
                                            self.yolov3_output_folder, self.yolo_epochs.text(),
                                            self.yolo_wEpochs.text(), self.yolo_initLR.text(),
                                            self.yolo_finalLR.text(),))
        self.action_process.start()
        self.run_tensorboard(self.yolov3_tensorboard_logs_folder)
        self.trainingTrackerLabel.setVisible(True)
        self.tensorboardIcon.setVisible(True)
        self.yoloStartTrainButton.setEnabled(False)

    def start_deepfill_train(self):
        self.action_process = Process(target=self.camouflage_api.on_deepfill_train_button_click,
                                      args=(self.deepfillv1_tensorboard_logs_folder,
                                            self.deepfillv1_output_folder,))
        self.action_process.start()
        self.run_tensorboard(self.deepfillv1_tensorboard_logs_folder)
        self.trainingTrackerLabel.setVisible(True)
        self.tensorboardIcon.setVisible(True)
        self.deepfillStartTrainButton.setEnabled(False)

    def stop_yolo_train(self):
        if self.action_process is not None:
            self.action_process.terminate()
            self.action_process = None
        if self.tensorboard_process is not None:
            self.tensorboard_process.kill()
            self.tensorboard_process = None
        self.yoloStartTrainButton.setEnabled(True)
        self.trainingTrackerLabel.setVisible(False)
        self.tensorboardIcon.setVisible(False)

    def stop_deepfill_train(self):
        if self.action_process is not None:
            self.action_process.terminate()
            self.action_process = None
        if self.tensorboard_process is not None:
            self.tensorboard_process.kill()
            self.tensorboard_process = None
        self.deepfillStartTrainButton.setEnabled(True)
        self.trainingTrackerLabel.setVisible(False)
        self.tensorboardIcon.setVisible(False)

    def run_tensorboard(self, log_dir):
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        self.tensorboard_process = subprocess.Popen(['tensorboard', '--logdir', log_dir])

    def yolo_train_parameter_validation(self):
        if self.yolo_batchSize.text() == '' or self.yolo_batchSize.text() is None:
            return False
        if self.yolo_initLR.text() == '' or self.yolo_initLR.text() is None:
            return False
        if self.yolo_finalLR.text() == '' or self.yolo_finalLR.text() is None:
            return False
        if self.yolo_wEpochs.text() == '' or self.yolo_wEpochs.text() is None:
            return False
        if self.yolo_epochs.text() == '' or self.yolo_epochs.text() is None:
            return False
        return True


if __name__ == '__main__':
    app = QApplication([])
    window = Window()
    window.show()
    sys.exit(app.exec_())
