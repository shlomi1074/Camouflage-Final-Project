from PIL import Image
import glob
image_list = []

newDir = 'E:\\FinalProject\\Datasets\\data\\AirshipsNew\\'
for filename in glob.glob('E:\\FinalProject\\Datasets\\data\\Airships\\*.JPEG'):
    image = Image.open(filename)
    new_image = image.resize((416, 416));
    name = filename.split("\\")
    new_image.save(newDir + name[-1]);
