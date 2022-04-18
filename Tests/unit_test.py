import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import unittest
import numpy as np
from Code.Models.YOLOV3.core.utils import read_class_names, get_anchors, image_preporcess, bboxes_iou


class Tests(unittest.TestCase):
    def test_read_class_names(self):
        path = r"../Code/Models/YOLOV3/data/classes/classes.names"
        class_name_list = read_class_names(path)
        self.assertEqual(len(class_name_list), 2, "Should be 2")

    def test_get_anchors(self):
        path = r"../Code/Models/YOLOV3/data/anchors/basline_anchors.txt"
        anchors_numpy_array = get_anchors(path)
        self.assertEqual(len(anchors_numpy_array), 3, "Should be 3")

    def test_image_preporcess_1(self):
        blank_image = np.zeros((500, 500, 3), np.uint8)
        resized_image = image_preporcess(blank_image, (416, 416))
        w, h, _ = resized_image.shape
        self.assertEqual(w, 416, "Should be 416")
        self.assertEqual(h, 416, "Should be 416")

    def test_image_preporcess_2(self):
        blank_image = np.zeros((400, 400, 3), np.uint8)
        resized_image = image_preporcess(blank_image, (416, 416))
        w, h, _ = resized_image.shape
        self.assertEqual(w, 416, "Should be 416")
        self.assertEqual(h, 416, "Should be 416")

    def test_image_preporcess_3(self):
        blank_image = np.zeros((350, 500, 3), np.uint8)
        resized_image = image_preporcess(blank_image, (416, 416))
        w, h, _ = resized_image.shape
        self.assertEqual(w, 416, "Should be 416")
        self.assertEqual(h, 416, "Should be 416")

    def test_bboxes_iou_1(self):
        boxes1 = np.array([105.68676758, 134.8099823, 290.64291382, 236.62156677])
        boxes2 = np.array([105.68676758, 134.8099823, 290.64291382, 236.62156677])
        iou = bboxes_iou(boxes1, boxes2)
        self.assertEqual(iou, 1, "Should be 1")

    def test_bboxes_iou_2(self):
        boxes1 = np.array([0, 0, 100, 100])
        boxes2 = np.array([0, 0, 100, 50])
        iou = bboxes_iou(boxes1, boxes2)
        self.assertEqual(iou, 0.5, "Should be 0.5")

    def test_bboxes_iou_3(self):
        boxes1 = np.array([0, 0, 100, 100])
        boxes2 = np.array([0, 0, 50, 50])
        iou = bboxes_iou(boxes1, boxes2)
        self.assertEqual(iou, 0.25, "Should be 0.25")


if __name__ == '__main__':
    unittest.main()
