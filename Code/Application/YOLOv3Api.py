import tensorflow as tf
#tf.compat.v1.enable_eager_execution()
import cv2
import core.utils as utils
import numpy as np
from core.yolov3 import YOLOv3, decode
from train import run

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

graph = None
class Yolov3Api:
    def __init__(self, input_size, iou_threshold):
        self.model_path = None
        self.model = None
        self.input_size = input_size
        self.iou_threshold = iou_threshold

    '''
    load existing model
    '''
    def load_model(self, yolov3_model_path):
        try:
            global graph
            self.model_path = yolov3_model_path
            # Build Model
            input_layer = tf.keras.layers.Input([self.input_size, self.input_size, 3])
            feature_maps = YOLOv3(input_layer)

            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                bbox_tensor = decode(fm, i)
                bbox_tensors.append(bbox_tensor)

            self.model = tf.keras.Model(input_layer, bbox_tensors)
            self.model.load_weights(self.model_path)
            graph = tf.get_default_graph()
            return True
        except Exception as e:
            print(e)
            self.model_path = None
            self.model = None
            return False
    '''
     Detect targets in a single image and return the bounding boxes of the targets
    '''
    def detect_target_bboxes(self, image_path):
        if self.model is None:
            return None

        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_size = image.shape[:2]
        image_data = utils.image_preporcess(np.copy(image), [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        global graph
        with graph.as_default():
            self.model._make_predict_function()
            pred_bbox = self.model.predict(image_data)
            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)
            bboxes = utils.postprocess_boxes(pred_bbox, image_size, self.input_size, self.iou_threshold)
            bboxes = utils.nms(bboxes, self.iou_threshold, method='nms')

        return image, bboxes

    '''
     Detect targets in a single image, draw the bounding boxes
      and save the image in a given location
    '''
    def detect_target_save(self, image_path, output_path):
        if self.model is None:
            return None
        #tf.compat.v1.disable_eager_execution()
        image_name = image_path.split('\\')[-1]
        image = self.detect_target_draw_bboxes(image_path)
        # cv2.imwrite(cfg.TEST.DECTECTED_IMAGE_PATH+image_name, image)
        cv2.imwrite(output_path + "\\" + image_name, image)
        #tf.compat.v1.enable_eager_execution()


    '''
     Detect targets in a single image, draw the bounding boxes
      and return the image with the bounding boxes
    '''
    def detect_target_draw_bboxes(self, image_path):
        if self.model is None:
            return None
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_size = image.shape[:2]
        image_data = utils.image_preporcess(np.copy(image), [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        pred_bbox = self.model.predict(image_data)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, image_size, self.input_size, self.iou_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold, method='nms')

        image = utils.draw_bbox(image, bboxes)
        return image

    def train_model(self, log_dir, output_dir, epochs, warmup_epochs, lr_init, end_lr):
        try:
            run(log_dir, output_dir, int(epochs), int(warmup_epochs), float(lr_init), float(end_lr))
        except Exception as e:
            print('Train failed:', e)


'''
for tests - needs to be removed 
'''
if __name__ == "__main__":
    pass
    # model_path = r'E:\FinalProject\Code\Models\YOLOV3\TrainedModel\tanks+airships\yolov3'
    # yolov3_api = Yolov3Api(416, 0.5)
    # #yolov3_api.load_model(model_path)
    # path = "E:\FinalProject\Datasets\data\Tanks\\n04389033_30632.JPEG"
    # out = r"E:\FinalProject\temp"
    #
    # #start = timer()
    # #yolov3_api.detect_target_save(path, out)
    # yolov3_api.train_model(r'./data/yolo/log', out)
    # #bboxes = yolov3_api.detect_target_bboxes(path)
    # #end = timer()
    # #print(end - start)
    # #print(single_image_detection_bboxs(path))