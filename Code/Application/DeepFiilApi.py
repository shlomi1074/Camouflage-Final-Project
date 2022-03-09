
#from train_inpainting import
from inpaint_model import InpaintCAModel
import neuralgym as ng
import cv2
import numpy as np
from PIL import Image
import YOLOv3Api
from timeit import default_timer as timer
import tensorflow as tf
from test_inpainting import fill
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class DeepFillApi:
    def __init__(self, input_size):
        self.model_path = None
        self.outputs = None
        self.image_placeholder = None
        self.model = None
        self.input_size = input_size
        # self.iou_threshold = iou_threshold

    def train_model(self, log_dir, output_dir):
        pass

    def generate_mask_image(self, bboxes):
        img = Image.open("./base.png")
        img = img.convert("RGBA")

        datas = img.getdata()
        newData = [[(0, 0, 0, 0)] * datas.size[0] for i in range(datas.size[1])]
        for bbox in bboxes:
            xmin = int(bbox[0]) - 1
            xmax = int(bbox[2]) + 1
            ymin = int(bbox[1]) - 1
            ymax = int(bbox[3]) + 1

            for j in range(ymin, ymax):
                for i in range(xmin, xmax):
                    newData[j][i] = (255, 255, 255, 255)

        flat = []
        for sub in newData:
            for item in sub:
                flat.append(item)

        img.putdata(flat)
        return img
        #img.save("./New.png", "PNG")

    def load_model(self, deepfill_model_dir):
        FLAGS = ng.Config('inpaint.yml')
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_config)

        model = InpaintCAModel()
        input_image_ph = tf.placeholder(
            tf.float32, shape=(1, self.input_size, self.input_size * 2, 3))
        output = model.build_server_graph(FLAGS, input_image_ph)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        self.outputs = output
        self.image_placeholder = input_image_ph
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(
                deepfill_model_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        self.model = sess

    def fill_image(self, image, bboxes):
        mask = self.generate_mask_image(bboxes)
        mask = mask.convert("RGB")
        mask = np.array(mask)
        assert image.shape == mask.shape

        h, w, _ = image.shape
        grid = 4
        image = image[:h // grid * grid, :w // grid * grid, :]
        mask = mask[:h // grid * grid, :w // grid * grid, :]

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)
        result = self.model.run(self.outputs, feed_dict={self.image_placeholder: input_image})
        return result
        #cv2.imwrite('.\\outp.png', result[0][:, :, ::-1])

'''
for tests - needs to be removed 
'''
if __name__ == "__main__":
    deepfill_model_path = r'E:\FinalProject\TrainedModels\release_places2_256_deepfill_v2'
    deppfill_api = DeepFillApi(416)
    deppfill_api.load_model(deepfill_model_path)
    yolo_model_path = r'E:\FinalProject\TrainedModels\YOLOv3\yolov3'
    yolov3_api = YOLOv3Api.Yolov3Api(416, 0.5)
    yolov3_api.load_model(yolo_model_path)

    #
    image_path = "E:\FinalProject\Datasets\data\Tanks\\n04389033_30632.JPEG"
    out = r"E:\FinalProject\temp"

    start = timer()
    #yolov3_api.detect_target_save(path, out)
    #yolov3_api.train_model(r'./data/log', out)
    image, bboxes = yolov3_api.detect_target_bboxes(image_path)
    end = timer()
    print(end - start)

    temp = deppfill_api.fill_image(image, bboxes)
    temp2 = deppfill_api.fill_image(image, bboxes)
    #print(single_image_detection_bboxs(path))