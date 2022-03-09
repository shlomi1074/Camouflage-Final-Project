import cv2
import os
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov3 import YOLOv3, decode


INPUT_SIZE   = 416
NUM_CLASS    = len(utils.read_class_names(cfg.YOLO.CLASSES))
CLASSES      = utils.read_class_names(cfg.YOLO.CLASSES)

def single_image_detection_save(image_path, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # Build Model
    input_layer  = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])
    feature_maps = YOLOv3(input_layer)

    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, i)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    model.load_weights("E:\\FinalProject\\Models\\YOLOV3\\yolov3")

    image_name = image_path.split('\\')[-1]
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_size = image.shape[:2]
    image_data = utils.image_preporcess(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    pred_bbox = model.predict(image_data)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bboxes = utils.postprocess_boxes(pred_bbox, image_size, INPUT_SIZE, cfg.TEST.SCORE_THRESHOLD)
    bboxes = utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method='nms')

    image = utils.draw_bbox(image, bboxes)
    # cv2.imwrite(cfg.TEST.DECTECTED_IMAGE_PATH+image_name, image)
    cv2.imwrite(output_dir + image_name, image)

def single_image_detection_bboxs(image_path):

    # Build Model
    input_layer = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])
    feature_maps = YOLOv3(input_layer)

    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, i)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    model.load_weights("E:\\FinalProject\\Models\\YOLOV3\\yolov3")

    image_name = image_path.split('\\')[-1]
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_size = image.shape[:2]
    image_data = utils.image_preporcess(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    pred_bbox = model.predict(image_data)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bboxes = utils.postprocess_boxes(pred_bbox, image_size, INPUT_SIZE, cfg.TEST.SCORE_THRESHOLD)
    bboxes = utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method='nms')

    return bboxes


if __name__ == "__main__":
    path = "E:\FinalProject\Datasets\data\Tanks\\n04389033_30632.JPEG"
    out = "E:\FinalProject\\temp\\"
    single_image_detection_save(path, out)
    #print(single_image_detection_bboxs(path))