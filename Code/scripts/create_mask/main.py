import cv2
from PIL import Image
from image_detection import single_image_detection_bboxs

def create_mask_image(bboxs):
    img = Image.open("./base.png")
    img = img.convert("RGBA")

    datas = img.getdata()
    import numpy as np
    print(datas.size[0])
    print(datas.size[1])
    newData = [[(0, 0, 0, 0)]*datas.size[0] for i in range(datas.size[1])]
    for bbox in bboxs:
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
    img.save("./New.png", "PNG")


if __name__ == "__main__":
    path = "E:\FinalProject\Datasets\data\Airships\\n02692877_16302.JPEG"
    out = "E:\FinalProject\\temp\\"
    bboxs = single_image_detection_bboxs(path)
    create_mask_image(bboxs)
    #print(single_image_detection_bboxs(path))