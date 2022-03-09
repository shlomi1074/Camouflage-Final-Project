import glob

from bs4 import BeautifulSoup

dstFile = 'airships_test.txt'
srcdir = 'E:\\FinalProject\\Datasets\\labels\\airships_test'
image_src_dir = 'E:\\FinalProject\\Datasets\\\data\\Airships'
w = open(srcdir + '\\' + dstFile, "a")
for filename in glob.glob(srcdir + '\\*.xml'):
    with open(filename, 'r') as f:
        data = f.read()
        Bs_data = BeautifulSoup(data, "xml")
        bboxs = Bs_data.find_all('bndbox')

    filename = image_src_dir + "\\" + filename.split("\\")[-1].split(".")[0] + ".JPEG"
    str = filename
    for bbox in bboxs:
        xmin = bbox.find('xmin').get_text()
        ymin = bbox.find('ymin').get_text()
        xmax = bbox.find('xmax').get_text()
        ymax = bbox.find('ymax').get_text()
        str = str + " " + xmin + "," + ymin + "," + xmax + "," + ymax + "," + "1"
        # print(image_src_dir + filename + " " + xmin + " " + ymin + " " + xmax + " " + ymax)
    w.write(str + "\n")
    print(str)

w.close()
f.close()
