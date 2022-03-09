import argparse
import os
from random import shuffle
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='./training_data', type=str,
                    help='The folder path')
parser.add_argument('--train_filename', default='./data_flist/train_shuffled.flist', type=str,
                    help='The train filename.')
parser.add_argument('--validation_filename', default='./data_flist/validation_shuffled.flist', type=str,
                    help='The validation filename.')
parser.add_argument('--is_shuffled', default='0', type=int,
                    help='Needed to be shuffled')

if __name__ == "__main__":

    args = parser.parse_args()

    # get the list of directories and separate them into 2 types: training and validation
   # training_dirs = os.listdir(args.folder_path + "/training")
    #validation_dirs = os.listdir(args.folder_path + "/validation")
    training_dirs = os.listdir("D:\\deepfill data\\train_large_places365standard\\data_large")
    validation_dirs = training_dirs
    # make 2 lists to save file paths
    training_file_names = []
    validation_file_names = []
    args.folder_path = 'D:\deepfill data\\train_large_places365standard\data_large'
    # append all files into 2 lists
    for training_dir in training_dirs:
        # append each file into the list file names
        training_folder = os.listdir(args.folder_path + "/" + training_dir)
        for training_item in training_folder:
            # modify to full path -> directory
            training_item = args.folder_path + "/" + training_dir + "/" + training_item
            i = 1
            for file in glob.glob(training_item + "\\*.jpg"):
                if i % 5 == 0:
                    validation_file_names.append(file)
                else:
                    training_file_names.append(file)
                i += 1

    # append all files into 2 lists
    # for validation_dir in validation_dirs:
    #     # append each file into the list file names
    #     validation_folder = os.listdir(args.folder_path + "/" + validation_dir)
    #     for validation_item in validation_folder:
    #         # modify to full path -> directory
    #         validation_item = args.folder_path + "/" + validation_dir + "/" + validation_item
    #         for file in glob.glob(validation_item  + "\\*.jpg"):
    #             validation_file_names.append(file)

    # print all file paths
    for i in training_file_names:
        print(i)
    for i in validation_file_names:
        print(i)

    # shuffle file names if set
    if args.is_shuffled == 1:
        shuffle(training_file_names)
        shuffle(validation_file_names)

    # make output file if not existed
    if not os.path.exists(args.train_filename):
        f = open(args.train_filename, "w")
        f.close()
        #os.mknod(args.train_filename)

    if not os.path.exists(args.validation_filename):
        f = open(args.validation_filename, "w")
        f.close()
        #os.mknod(args.validation_filename)

    # write to file
    fo = open(args.train_filename, "w")
    fo.write("\n".join(training_file_names))
    fo.close()

    fo = open(args.validation_filename, "w")
    fo.write("\n".join(validation_file_names))
    fo.close()

    # print process
    print("Written file is: ", args.train_filename, ", is_shuffle: ", args.is_shuffled)