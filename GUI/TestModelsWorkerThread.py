import os

from PyQt5 import QtCore
import time
from ButtonCommands import GuiFunctions
from timeit import default_timer as timer


class TestModelsThread(QtCore.QThread):

    results = QtCore.pyqtSignal(object)

    def __init__(self, q):
        QtCore.QThread.__init__(self)
        self.queue = q
        self.is_yolo_loaded = False
        self.is_deepfill_loaded = True
        self.camouflage_api = GuiFunctions(416, 0.5)

    def run(self):
        """
        1 - load yolo
        2 - load deepfill
        3 - single test
        4 - batch test
        5 - model status
        """
        while True:
            item = self.queue.get()
            if item[0] == 1:
                self.results.emit([3, r'Loading YOLOv3...'])
                self.load_yolo_model(item[1])
                self.results.emit([1, self.is_yolo_loaded])
                self.results.emit([4, 'yolo', self.is_yolo_loaded])
                time.sleep(2)
            if item[0] == 2:
                self.results.emit([3, r'Loading DeepFillv1...'])
                self.load_deepfill_model(item[1])
                self.results.emit([2, self.is_deepfill_loaded])
                self.results.emit([4, 'deepfill', self.is_deepfill_loaded])
                time.sleep(2)
            if item[0] == 3:
                self.single_image_test(item[1])
            if item[0] == 4:
                self.batch_test(item[1], item[2])
            if item[0] == 5:
                self.results.emit([4, 'yolo', self.is_yolo_loaded])
                self.results.emit([4, 'deepfill', self.is_deepfill_loaded])


            time.sleep(1)

    def load_yolo_model(self, model_path):
        res = self.camouflage_api.load_yolov3_model(model_path)
        self.is_yolo_loaded = res
        print("Yolo loaded result ", self.is_yolo_loaded)

    def load_deepfill_model(self, model_path):
        res = self.camouflage_api.load_deepfill_model(model_path)
        self.is_deepfill_loaded = res
        print("deepfill loaded result ", res)

    def single_image_test(self, image_path):
        if self.is_deepfill_loaded and self.is_yolo_loaded:
            try:
                self.camouflage_api.on_single_test_button_click(image_path)
                self.results.emit([1, True, 'Success'])
            except Exception as e:
                self.results.emit([1, False, e])
        else:
            self.results.emit([1, False, 'Models are not loaded'])

    def batch_test(self, input_dir, output_dir):
        if self.is_deepfill_loaded and self.is_yolo_loaded:
            start = timer()
            i = 0
            self.results.emit([2, 'text', r"========== Start batch test =========="])

            total_files = len([name for name in os.listdir(input_dir) if
                               os.path.isfile(os.path.join(input_dir, name))])
            self.results.emit([2, 'text', f"Total files detected: {total_files}\n"])

            for filename in os.listdir(input_dir):
                file = os.path.join(input_dir, filename)
                # checking if it is a file
                if os.path.isfile(file):
                    i += 1
                    try:
                        self.results.emit([2, 'text', f"{i} currently testing file: {filename}"])
                        self.camouflage_api.on_batch_test_button_click(file, input_dir, output_dir)
                        self.results.emit([2, 'text', f"{i} finished testing file: {filename} successfully"])

                        self.results.emit([2, 'list', output_dir + '\\' + file.split("\\")[-1]])

                    except:
                        self.results.emit([2, 'text', f"{i} {filename} failed."])

            end = timer()
            self.results.emit([2, 'text', f"Testing time of {i} files: {end - start}"])
            self.results.emit([2, 'text', f"Average time per image: {(end - start)/i}"])



