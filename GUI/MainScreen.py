import os
import shutil
from multiprocessing import Process
import subprocess
from PyQt5 import uic, QtCore, QtWidgets
from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QLabel
import pandas as pd
import TrainLoggerThread
from ButtonCommands import GuiFunctions
from MplCanvas import MplCanvas
from calculate_map import run


class Window(QMainWindow):
    def __init__(self, worker_thread, q):
        super().__init__()
        # DEFAULT VALUES
        self.log_path = None
        self.train_log = None
        self.yolov3_output_folder = r'C:\trainedModels\yolov3'
        self.yolov3_tensorboard_logs_folder = r'C:\tensorboard\logs\yolov3'
        self.deepfillv1_output_folder = r'C:\trainedModels\deepfillv1'
        self.deepfillv1_tensorboard_logs_folder = r'C:\tensorboard\logs\deepfillv1'
        self.yolo_model_path = r'..\FinalProject\TrainedModels\YOLOv3\yolov3'
        self.deepfill_model_path = r'..\FinalProject\TrainedModels\release_places2_256_deepfill_v2'
        self.batch_test_source_folder_path = r'Select source folder'
        self.batch_test_output_folder_path = r'Select output folder'
        self.batch_test_preview_list = []
        self.batch_test_original_list = []
        self.batch_test_preview_index = 0

        self.camouflage_api = GuiFunctions(416, 0.5)
        self.workerThread = worker_thread
        self.queue = q
        self.workerThread.results.connect(self.worker_event)

        self.action_process = None
        self.action_process_2 = None
        self.tensorboard_process = None
        self.is_yolo_loaded = False
        self.is_deepfill_loaded = False
        self.is_training_results = True
        self.charts = None
        self.training_data = None
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
        self.resultsToggleButton.setVisible(False)
        self.yoloModelPath.setText(self.yolo_model_path)
        self.deepfillModelPath.setText(self.deepfill_model_path)
        self.yoloModelPathBatchTest.setText(self.yolo_model_path)
        self.deepfillModelPathBatchTest.setText(self.deepfill_model_path)
        self.batchTestSourcePath.setText(self.batch_test_source_folder_path)
        self.batchTestOutputPath.setText(self.batch_test_output_folder_path)

        """
        click events
        """
        self.yoloStartTrainButton.clicked.connect(self.start_yolo_train)
        self.yoloStopTrainButton.clicked.connect(self.stop_yolo_train)
        self.deepfillStartTrainButton.clicked.connect(self.start_deepfill_train)
        self.deepfillStopTrainButton.clicked.connect(self.stop_deepfill_train)
        self.saveModelButton.mousePressEvent = self.yolov3_select_model_folder
        self.saveModelButton_2.mousePressEvent = self.deepfillv1_select_model_folder
        self.SelectTestImageButton.mousePressEvent = self.test_single_image
        self.tensorboardLogFolderButton.mousePressEvent = self.yolov3_select_logs_folder
        self.tensorboardLogFolderButton_2.mousePressEvent = self.deepfillv1_select_logs_folder
        self.LoadDeepfillModelButton.mousePressEvent = self.deepfill_select_model_path
        self.LoadDeepfillModelBatchTestButton.mousePressEvent = self.deepfill_select_model_path
        self.LoadYoloModelButton.mousePressEvent = self.yolo_select_model_path
        self.LoadYoloModelBatchTestButton.mousePressEvent = self.yolo_select_model_path
        self.tabContainer.currentChanged.connect(self.on_tab_changes)
        self.batchTestSourceFolderButton.mousePressEvent = self.batch_test_select_source_folder
        self.batchTestOutputFolderButton.mousePressEvent = self.batch_test_select_output_folder
        self.TensorboardFileButton.mousePressEvent = self.load_tensor_report
        self.startBatchTest.clicked.connect(self.start_batch_test)
        self.previewNext.clicked.connect(self.batch_test_preview_next_button)
        self.previewPrev.clicked.connect(self.batch_test_preview_prev_button)
        self.resultsToggleButton.clicked.connect(self.results_toggle)
        self.saveFinalOutButton.clicked.connect(self.save_single_test_output)
        self.excel_button.mousePressEvent = self.excel_export_button_click
        #run()  # TODO: RUN IT AFTER YOLO TRAINING

    def closeEvent(self, event):
        print('close event')
        folder = 'temp'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                event.accept()
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
                event.accept()

    def on_tab_changes(self, selected_index):
        pass
        #if selected_index == 2:
            # self.gridLayout.removeWidget(self.charts)
            # self.charts = MplCanvas(parent=self, data_path=
            # r"E:\FinalProject\backup_files\events.out.tfevents.1646244711.SHLOMI-PC.18044.0.v2")
            # self.gridLayout.addWidget(self.charts)
        if selected_index == 3:
            self.queue.put([5])
        if selected_index == 4:
            self.queue.put([5])

    def results_toggle(self):
        if self.is_training_results:
            self.is_training_results = False
            self.resultsToggleButton.setText("View Training Results")
            #run()
            self.gridLayout.setHorizontalSpacing(2)
            self.gridLayout.setVerticalSpacing(2)

            while self.gridLayout.count():
                item = self.gridLayout.takeAt(0)
                widget = item.widget()
                widget.deleteLater()
            mAP = QLabel('mAP graph will be here', self)
            mAP.setPixmap(QPixmap(r'..\Code\Models\mAP\results\mAP.png'))
            mAP.setScaledContents(True)

            GT = QLabel('GT graph will be here', self)
            GT.setPixmap(QPixmap(r'..\Code\Models\mAP\results\Ground-Truth Info.png'))
            GT.setScaledContents(True)

            PO = QLabel('PO graph will be here', self)
            PO.setPixmap(QPixmap(r'..\Code\Models\mAP\results\Predicted Objects Info.png'))
            PO.setScaledContents(True)

            Tanks = QLabel('Tanks graph will be here', self)
            Tanks.setPixmap(QPixmap(r'..\Code\Models\mAP\results\classes\Tank.png'))
            Tanks.setScaledContents(True)

            Airship = QLabel('Tanks graph will be here', self)
            Airship.setPixmap(QPixmap(r'..\Code\Models\mAP\results\classes\Airship.png'))
            Airship.setScaledContents(True)

            self.gridLayout.addWidget(mAP, 1, 1)
            self.gridLayout.addWidget(GT, 1, 2)
            self.gridLayout.addWidget(PO, 1, 3)
            self.gridLayout.addWidget(Tanks, 2, 1)
            self.gridLayout.addWidget(Airship, 2, 2)
        else:
            self.is_training_results = True
            self.resultsToggleButton.setText("View Validation Results")

            while self.gridLayout.count():
                item = self.gridLayout.takeAt(0)
                widget = item.widget()
                widget.deleteLater()
            if self.log_path is not None:
                self.load_tensor_report_from_log(self.log_path)

    def worker_event(self, event_data):
        """
        1 - single image test
        2 - batch test
        3 -
        4 -
        """
        if event_data[0] == 1:
            if event_data[1]:
                self.original_img.setPixmap(QPixmap(r'temp\orig.png'))
                self.yolo_img.setPixmap(QPixmap(r'temp\yolo.png'))
                self.mask_img.setPixmap(QPixmap(r'temp\mask.png'))
                self.deepfill_img.setPixmap(QPixmap(r'temp\deepfill.png'))

        if event_data[0] == 2:
            if event_data[1] == 'text':
                self.batchTestingProcess.append(event_data[2])
            if event_data[1] == 'list':
                self.batch_test_preview_list.append(event_data[2])
                self.batch_test_original_list.append(event_data[3])

            if len(self.batch_test_preview_list) > 0:
                self.previewImage.setPixmap(QPixmap(self.batch_test_preview_list[0]))
                self.originalImage.setPixmap(QPixmap(self.batch_test_original_list[0]))


        if event_data[0] == 4:
            if event_data[1] == 'yolo':
                self.is_yolo_loaded = event_data[2]
                if event_data[2]:
                    self.yoloModelStatus.setPixmap(QPixmap(r'Resources\green.png'))
                    self.yoloModelStatus_2.setPixmap(QPixmap(r'Resources\green.png'))
                else:
                    self.yoloModelStatus.setPixmap(QPixmap(r'Resources\red.png'))
                    self.yoloModelStatus_2.setPixmap(QPixmap(r'Resources\red.png'))
            if event_data[1] == 'deepfill':
                self.is_deepfill_loaded = event_data[2]
                if event_data[2]:
                    self.deepfillModelStatus.setPixmap(QPixmap(r'Resources\green.png'))
                    self.deepfillModelStatus_2.setPixmap(QPixmap(r'Resources\green.png'))
                else:
                    self.deepfillModelStatus.setPixmap(QPixmap(r'Resources\red.png'))
                    self.deepfillModelStatus_2.setPixmap(QPixmap(r'Resources\red.png'))

    def save_single_test_output(self):
        try:
            dst_dir = QFileDialog.getExistingDirectory(self, 'Select Destination Folder')
            if dst_dir != '' and dst_dir is not None:
                shutil.copy(r'temp\orig.png', dst_dir)
                shutil.copy(r'temp\yolo.png', dst_dir)
                shutil.copy(r'temp\mask.png', dst_dir)
                shutil.copy(r'temp\deepfill.png', dst_dir)
        except:
            print('save images failed')

    def load_tensor_report(self, event):
        self.log_path, _ = QFileDialog.getOpenFileName(self, 'Select log file')
        if self.log_path != '' and self.log_path is not None:
            self.load_tensor_report_from_log(self.log_path)

    def load_tensor_report_from_log(self, log_path):
        #self.gridLayout.removeWidget(self.charts)
        if self.charts is not None:
            self.is_training_results = True
            self.resultsToggleButton.setText("View Validation Results")

            while self.gridLayout.count():
                item = self.gridLayout.takeAt(0)
                widget = item.widget()
                widget.deleteLater()
        self.charts = MplCanvas(parent=self, data_path=log_path)
        self.gridLayout.addWidget(self.charts)

    def batch_test_select_source_folder(self, event):
        src_dir = QFileDialog.getExistingDirectory(self, 'Select Source Folder')
        if src_dir != '' and src_dir is not None:
            self.batch_test_source_folder_path = src_dir
        self.batchTestSourcePath.setText(self.batch_test_source_folder_path)

    def batch_test_select_output_folder(self, event):
        dst_dir = QFileDialog.getExistingDirectory(self, 'Select Source Folder')
        if dst_dir != '' and dst_dir is not None:
            self.batch_test_output_folder_path = dst_dir
        self.batchTestOutputPath.setText(self.batch_test_output_folder_path)

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

    def test_single_image(self, event):
        if self.is_yolo_loaded and self.is_deepfill_loaded:
            image_path, _ = QFileDialog.getOpenFileName(self, 'Select Image', 'c:\\', "Image files (*.jpg *.png *.jpeg)")
            if image_path != '' and image_path is not None:
                movie = QMovie(r"Resources\nspinner4.gif")
                self.original_img.setPixmap(QPixmap(image_path))
                self.yolo_img.setMovie(movie)
                self.mask_img.setMovie(movie)
                self.deepfill_img.setMovie(movie)
                movie.start()
                self.queue.put([3, image_path])

    def deepfill_select_model_path(self, event):
        deepfill_model_dir = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if deepfill_model_dir != '' and deepfill_model_dir is not None:
            self.deepfill_model_path = deepfill_model_dir
            self.deepfillModelPath.setText(self.deepfill_model_path)
            self.deepfillModelPathBatchTest.setText(self.deepfill_model_path)
            self.deepfillModelStatus.setPixmap(QPixmap(r'Resources\red.png'))
            self.deepfillModelStatus_2.setPixmap(QPixmap(r'Resources\red.png'))
            self.queue.put([2, deepfill_model_dir])

    def yolo_select_model_path(self, event):
        yolo_model_dir = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if yolo_model_dir != '' and yolo_model_dir is not None:
            self.yolo_model_path = yolo_model_dir
            self.yoloModelPath.setText(self.yolo_model_path)
            self.yoloModelPathBatchTest.setText(self.yolo_model_path)
            self.yoloModelStatus.setPixmap(QPixmap(r'Resources\red.png'))
            self.yoloModelStatus_2.setPixmap(QPixmap(r'Resources\red.png'))
            self.queue.put([1, yolo_model_dir + r'\yolov3'])

    def start_batch_test(self):
        if self.is_yolo_loaded and self.is_deepfill_loaded:
            self.batchTestingProcess.clear()
            self.batch_test_preview_list = []
            self.batch_test_original_list = []
            self.batch_test_preview_index = 0
            movie = QMovie(r"Resources\nspinner4.gif")
            self.previewImage.setMovie(movie)
            self.originalImage.setMovie(movie)
            movie.start()
            self.queue.put([4, self.batch_test_source_folder_path, self.batch_test_output_folder_path])

    def batch_test_preview_next_button(self):
        if self.batch_test_preview_index >= len(self.batch_test_preview_list):
            return
        try:
            self.previewImage.setPixmap(QPixmap(self.batch_test_preview_list[self.batch_test_preview_index + 1]))
            self.originalImage.setPixmap(QPixmap(self.batch_test_original_list[self.batch_test_preview_index + 1]))
            self.batch_test_preview_index += 1
        except:
            print('No more images')

    def batch_test_preview_prev_button(self):
        if self.batch_test_preview_index <= 0:
            return
        try:
            self.previewImage.setPixmap(QPixmap(self.batch_test_preview_list[self.batch_test_preview_index - 1]))
            self.originalImage.setPixmap(QPixmap(self.batch_test_original_list[self.batch_test_preview_index - 1]))
            self.batch_test_preview_index -= 1
        except:
            print('No more images')

    def log_training_process(self, log_data):
        self.trainingProcessText.setText(log_data)
        self.trainingProcessText.verticalScrollBar().setValue(self.trainingProcessText.verticalScrollBar().maximum())

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
        self.train_log = TrainLoggerThread.TrainLoggerThread()
        self.train_log.log_data.connect(self.log_training_process)
        self.train_log.start()

        self.trainingTrackerLabel.setVisible(True)
        self.tensorboardIcon.setVisible(True)

        self.yoloStartTrainButton.setEnabled(False)
        self.deepfillStartTrainButton.setEnabled(False)
        self.deepfillStopTrainButton.setEnabled(False)

    def start_deepfill_train(self):
        self.action_process = Process(target=self.camouflage_api.on_deepfill_train_button_click,
                                      args=(self.deepfillv1_tensorboard_logs_folder,
                                            self.deepfillv1_output_folder,))
        self.action_process.start()
        self.run_tensorboard(self.deepfillv1_tensorboard_logs_folder)
        self.train_log = TrainLoggerThread.TrainLoggerThread()
        self.train_log.log_data.connect(self.log_training_process)
        self.train_log.start()
        self.trainingTrackerLabel.setVisible(True)
        self.tensorboardIcon.setVisible(True)
        self.deepfillStartTrainButton.setEnabled(False)
        self.yoloStartTrainButton.setEnabled(False)
        self.yoloStopTrainButton.setEnabled(False)

    def stop_yolo_train(self):
        if self.action_process is not None:
            self.action_process.terminate()
            self.action_process = None
        if self.tensorboard_process is not None:
            self.tensorboard_process.kill()
            self.tensorboard_process = None
        self.yoloStartTrainButton.setEnabled(True)
        self.deepfillStartTrainButton.setEnabled(True)
        self.deepfillStopTrainButton.setEnabled(True)
        self.yoloStopTrainButton.setEnabled(True)
        self.trainingTrackerLabel.setVisible(False)
        self.tensorboardIcon.setVisible(False)
        self.train_log.terminate()

    def stop_deepfill_train(self):
        if self.action_process is not None:
            self.action_process.terminate()
            self.action_process = None
        if self.tensorboard_process is not None:
            self.tensorboard_process.kill()
            self.tensorboard_process = None
        self.yoloStartTrainButton.setEnabled(True)
        self.deepfillStartTrainButton.setEnabled(True)
        self.deepfillStopTrainButton.setEnabled(True)
        self.yoloStopTrainButton.setEnabled(True)
        self.trainingTrackerLabel.setVisible(False)
        self.tensorboardIcon.setVisible(False)
        self.train_log.terminate()

    def run_tensorboard(self, log_dir):
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        self.tensorboard_process = subprocess.Popen(['tensorboard', '--logdir', log_dir])
        #os.system(r"tensorboard --logdir " + log_dir)

    def excel_export_button_click(self, event):
        dst_dir = QFileDialog.getExistingDirectory(self, 'Select Destination Folder')
        if dst_dir != '' and dst_dir is not None:
            try:
                self.excel_export(dst_dir, self.training_data)
            except Exception as e:
                print(e)

    def excel_export(self, output_path, data):
        df = pd.DataFrame(data)
        writer = pd.ExcelWriter(output_path + r'/training_report.xlsx', engine='xlsxwriter')
        df.to_excel(writer, sheet_name='training_report', index=False)
        writer.save()

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