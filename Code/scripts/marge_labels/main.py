import random

src_dir = 'E:\FinalProject\Models\YOLOV3\data\dataset\\'
file_list = [src_dir + 'airships_test.txt', src_dir + 'tanks_test.txt']

i = 0;
merged_list = []
for file in file_list:
    my_file = open(file, "r")
    for line in my_file.readlines():
        i += 1
        merged_list.append(line)
    my_file.close()

random.shuffle(merged_list)
random.shuffle(merged_list)
random.shuffle(merged_list)

output_file = open(src_dir + 'annotation_test.txt', "a")

for line in merged_list:
    print(line)
    output_file.write(line)

output_file.close()