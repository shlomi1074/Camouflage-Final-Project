import sys
import os
import shutil
from multiprocessing import Process
import subprocess
import threading
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from ButtonCommands import GuiFunctions
from timeit import default_timer as timer
from MplCanvas import MplCanvas


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        # DEFAULT VALUES
        self.yolov3_output_folder = r'C:\trainedModels\yolov3'
        self.yolov3_tensorboard_logs_folder = r'C:\tensorboard\logs\yolov3'
        self.deepfillv1_output_folder = r'C:\trainedModels\deepfillv1'
        self.deepfillv1_tensorboard_logs_folder = r'C:\tensorboard\logs\deepfillv1'
        self.yolo_model_path = r'E:\FinalProject\TrainedModels\YOLOv3\yolov3'
        self.deepfill_model_path = r'E:\FinalProject\TrainedModels\release_places2_256_deepfill_v2'
        self.batch_test_source_folder_path = r'Select source folder'
        self.batch_test_output_folder_path = r'Select output folder'
        self.batch_test_preview_list = []
        self.batch_test_preview_index = 0

        self.camouflage_api = GuiFunctions(416, 0.5)
        self.action_process = None
        self.action_process_2 = None
        self.tensorboard_process = None
        self.is_yolo_loaded = False
        self.is_deepfill_loaded = False
        self.charts = None
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

    def on_tab_changes(self, selected_index):
        #if selected_index == 2:
            # self.gridLayout.removeWidget(self.charts)
            # self.charts = MplCanvas(parent=self, data_path=
            # r"E:\FinalProject\backup_files\events.out.tfevents.1646244711.SHLOMI-PC.18044.0.v2")
            # self.gridLayout.addWidget(self.charts)
        if selected_index == 3:
            if not self.is_yolo_loaded or not self.is_deepfill_loaded:
                #self.load_models()
                load_models_thread = threading.Thread(target=self.load_models)
                load_models_thread.start()

    def load_tensor_report(self, event):
        log_path, _ = QFileDialog.getOpenFileName(self, 'Select log file')
        if log_path != '' and log_path is not None:
            self.load_tensor_report_from_log(log_path)

    def load_tensor_report_from_log(self, log_path):
        self.gridLayout.removeWidget(self.charts)
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
        image_path, _ = QFileDialog.getOpenFileName(self, 'Select Image', 'c:\\', "Image files (*.jpg *.png *.jpeg)")
        if image_path != '' and image_path is not None:
            movie = QMovie(r"E:\FinalProject\GUI\Resources\nspinner4.gif")
            self.original_img.setPixmap(QPixmap(image_path))
            self.yolo_img.setMovie(movie)
            self.mask_img.setMovie(movie)
            self.deepfill_img.setMovie(movie)
            movie.start()
            t1 = threading.Thread(target=self.single_image_test_thread,
                                  args=(image_path,))
            t1.start()

    def single_image_test_thread(self, image_path):
        print("start image thread 1")
        self.load_deepfill_model(self.deepfill_model_path)
        self.load_yolo_model(self.yolo_model_path)
        self.camouflage_api.on_single_test_button_click(image_path)
        print("end image thread 1")
        self.original_img.setPixmap(QPixmap(r'temp\orig.png'))
        self.yolo_img.setPixmap(QPixmap(r'temp\yolo.png'))
        self.mask_img.setPixmap(QPixmap(r'temp\mask.png'))
        self.deepfill_img.setPixmap(QPixmap(r'temp\deepfill.png'))

    def deepfill_select_model_path(self, event):
        deepfill_model_dir = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if deepfill_model_dir != '' and deepfill_model_dir is not None:
            self.deepfill_model_path = deepfill_model_dir
            self.deepfillModelPath.setText(self.deepfill_model_path)
            self.deepfillModelPathBatchTest.setText(self.deepfill_model_path)
            self.load_models()

    def yolo_select_model_path(self, event):
        yolo_model_dir = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if yolo_model_dir != '' and yolo_model_dir is not None:
            self.yolo_model_path = yolo_model_dir
            self.yoloModelPath.setText(self.yolo_model_path)
            self.yoloModelPathBatchTest.setText(self.yolo_model_path)
            self.load_models()

    def load_models(self):
        pass
        # print("loading models...")
        # self.load_deepfill_model(self.deepfill_model_path)
        # self.load_yolo_model(self.yolo_model_path)
        # t1 = threading.Thread(target=self.load_deepfill_model,
        #                       args=(self.deepfill_model_path,))
        # t1.start()
        # t1.join()
        # t2 = threading.Thread(target=self.load_yolo_model,
        #                       args=(self.yolo_model_path,))
        # # starting thread 1
        # # starting thread 2
        # t2.start()
        # t2.join()

    def load_yolo_model(self, model_path):
        res = self.camouflage_api.load_yolov3_model(model_path)
        self.is_yolo_loaded = res
        print("Yolo loaded result ", self.is_yolo_loaded)

    def load_deepfill_model(self, model_path):
        res = self.camouflage_api.load_deepfill_model(model_path)
        self.is_deepfill_loaded = res
        print("deepfill loaded result ", res)

    def start_single_image_test(self, image_path):
        print(image_path)
        if image_path != '' and image_path is not None:
            orig_image, yolo_image, mask_image, deepfill_image = self.camouflage_api.on_single_test_button_click(image_path)

    def start_batch_test(self):
        self.batchTestingProcess.clear()
        self.batch_test_preview_list = []
        self.batch_test_preview_index = 0
        movie = QMovie(r"E:\FinalProject\GUI\Resources\nspinner4.gif")
        self.previewImage.setMovie(movie)
        movie.start()
        t1 = threading.Thread(target=self.start_batch_test_thread)
        t1.start()

    def start_batch_test_thread(self):
        self.batchTestingProcess.append("========== Loading models ==========")
        start = timer()
        self.batchTestingProcess.append("Loading Deepfill model...")

        self.load_deepfill_model(self.deepfill_model_path)
        self.batchTestingProcess.append(f"Deepfill loading result: {self.is_deepfill_loaded}")

        self.batchTestingProcess.append("Loading Yolov3 model...")

        self.load_yolo_model(self.yolo_model_path)
        self.batchTestingProcess.append(f"Yolov3 loading result: {self.is_yolo_loaded}")

        end = timer()
        self.batchTestingProcess.append(f"Loading models finished. Total time: {end - start}\n\n")

        start = timer()
        # self.camouflage_api.on_batch_test_button_click(self.batch_test_source_folder_path,
        #                                                self.batch_test_output_folder_path)
        i = 0
        self.batchTestingProcess.append("========== Start batch test ==========")
        total_files = len([name for name in os.listdir(self.batch_test_source_folder_path) if
                           os.path.isfile(os.path.join(self.batch_test_source_folder_path, name))])
        self.batchTestingProcess.append(f"Total files detected: {total_files}\n")

        for filename in os.listdir(self.batch_test_source_folder_path):
            file = os.path.join(self.batch_test_source_folder_path, filename)
            # checking if it is a file
            if os.path.isfile(file):
                i += 1
                try:
                    self.batchTestingProcess.append(f"{i} currently testing file: {filename}")
                    self.camouflage_api.on_batch_test_button_click(file, self.batch_test_source_folder_path,
                                                                   self.batch_test_output_folder_path)
                    self.batchTestingProcess.append(f"{i} finished testing file: {filename} successfully")
                    self.batch_test_preview_list.append(self.batch_test_output_folder_path + '\\' + file.split("\\")[-1])

                except:
                    self.batchTestingProcess.append(f"{i} {filename} failed.")

        end = timer()
        self.batchTestingProcess.append(f"Testing time of {i} files: {end - start}")
        self.batchTestingProcess.append(f"Average time per image: {(end - start)/i}")
        if len(self.batch_test_preview_list) > 0:
            self.previewImage.setPixmap(QPixmap(self.batch_test_preview_list[0]))

    def batch_test_preview_next_button(self):
        if self.batch_test_preview_index >= len(self.batch_test_preview_list):
            return
        try:
            self.previewImage.setPixmap(QPixmap(self.batch_test_preview_list[self.batch_test_preview_index + 1]))
            self.batch_test_preview_index += 1
        except:
            print('failed')

    def batch_test_preview_prev_button(self):
        if self.batch_test_preview_index <= 0:
            return
        try:
            self.previewImage.setPixmap(QPixmap(self.batch_test_preview_list[self.batch_test_preview_index - 1]))
            self.batch_test_preview_index -= 1
        except:
            print('failed')

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
        self.deepfillStartTrainButton.setEnabled(False)
        self.deepfillStopTrainButton.setEnabled(False)

    def start_deepfill_train(self):
        self.action_process = Process(target=self.camouflage_api.on_deepfill_train_button_click,
                                      args=(self.deepfillv1_tensorboard_logs_folder,
                                            self.deepfillv1_output_folder,))
        self.action_process.start()
        self.run_tensorboard(self.deepfillv1_tensorboard_logs_folder)
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