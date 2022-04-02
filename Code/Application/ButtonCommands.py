import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import YOLOv3Api
import DeepFiilApi
import cv2
import tensorflow as tf  # tf.compat.v1.enable_eager_execution()
from tensorflow.python.eager.context import eager_mode  # , graph_mode

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# with graph_mode(): with eager_mode():

class GuiFunctions:
    def __init__(self, input_size, iou_threshold):
        self.yolov3_api = YOLOv3Api.Yolov3Api(input_size, iou_threshold)
        self.deepfill_api = DeepFiilApi.DeepFillApi(input_size)

    def load_yolov3_model(self, yolov3_model_path):
        result = self.yolov3_api.load_model(yolov3_model_path)
        return result

    def load_deepfill_model(self, deepfill_model_path):
        try:
            self.deepfill_api.load_model(deepfill_model_path)
            return True
        except Exception as e:
            print(e)
            return False

    def on_yolov3_train_button_click(self, log_dir, output_path, epochs, warmup_epochs, lr_init, end_lr):
        with eager_mode():
            try:
                epochs = int(epochs)
                warmup_epochs = int(warmup_epochs)
                lr_init = float(lr_init)
                end_lr = float(end_lr)
                self.yolov3_api.train_model(log_dir, output_path, epochs, warmup_epochs, lr_init, end_lr)
                from calculate_map import run
                run()
            except ValueError:
                print("Please fill valid parameters")

    def on_deepfill_train_button_click(self, log_dir, output_path):
        path = 'output.txt'
        f = open(path, 'w')
        f.close()
        self.deepfill_api.train_model(log_dir, output_path)

        # try:
        #     self.deepfill_api.train_model(log_dir, output_path)
        # except:
        #     print("Oops!", sys.exc_info()[0], "occurred.")

    def on_batch_test_button_click(self, file, input_dir_path, output_dir_path):
        image, bboxes = self.yolov3_api.detect_target_bboxes(file)
        result_image = self.deepfill_api.fill_image(image, bboxes)
        cv2.imwrite(output_dir_path + '\\' + file.split("\\")[-1], result_image[0][:, :, ::-1])

    def on_single_test_button_click(self, image_path):
        img = cv2.imread(image_path)
        image, bboxes = self.yolov3_api.detect_target_bboxes(image_path)
        yolov3_image = self.yolov3_api.detect_target_draw_bboxes(image_path)
        mask = self.deepfill_api.paint_mask_on_image(image_path, bboxes)
        deepfill_output = self.deepfill_api.fill_image(image, bboxes)

        cv2.imwrite(r'.\temp\orig.png', img)
        cv2.imwrite(r'.\temp\yolo.png', yolov3_image)
        cv2.imwrite(r'.\temp\mask.png', mask)
        cv2.imwrite(r'.\temp\deepfill.png', deepfill_output[0][:, :, ::-1])
        return img, yolov3_image, mask, deepfill_output


if __name__ == "__main__":
    pass
    # functions = GuiFunctions(416, 0.5)
    # deep_res = functions.load_deepfill_model(r'E:\FinalProject\TrainedModels\release_places2_256_deepfill_v2')
    # print(deep_res)
    # yolo_res = functions.load_yolov3_model(r'E:\FinalProject\TrainedModels\YOLOv3\yolov3')
    # print(yolo_res)
    # # functions.on_batch_test_button_click(r'E:\FinalProject\Datasets\test_batch', r'E:\FinalProject\temp')
    # # functions.on_deepfill_train_button_click('data/log', r'E:\FinalProject\temp')
    # # i, y,m,d = functions.on_single_test_button_click(r"E:\FinalProject\Datasets\test_batch\n02692877_8463.JPEG")
    # i, y, m, d = functions.on_single_test_button_click(r"E:\FinalProject\Datasets\data\Tanks\\n04389033_30632.JPEG")
    # # functions.yolov3_api.detect_target_bboxes(image_path="E:\FinalProject\Datasets\data\Tanks\\n04389033_30632.JPEG")
    # cv2.imwrite(r'E:\FinalProject\temp\orig.png', i)
    # cv2.imwrite(r'E:\FinalProject\temp\yolo.png', y)
    # cv2.imwrite(r'E:\FinalProject\temp\mask.png', m)
    # cv2.imwrite(r'E:\FinalProject\temp\deep.png', d[0][:, :, ::-1])
    # functions.on_deepfill_train_button_click('data/log', r'E:\FinalProject\temp')
    # functions.on_yolov3_train_button_click('data/log', r'E:\FinalProject\temp')

