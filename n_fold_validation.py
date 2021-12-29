import sys
import numpy as np
import random
from random import randrange

SEED = 66

random.seed(SEED)

# Number of Folds
FOLD = 5

# read image
def read_train_image(path):
    dataset = []
    with open(path,"r") as f:
        for line in f.readlines():
            feature = line.split()
            feature = list(map(float, feature))
            dataset.append(feature)
    return np.array(dataset)

# read label
def read_train_label(path):
    label = []
    with open(path,"r") as f:
        for line in f.readlines():
            label.append(int(line))
    return np.array(label) 

# write label txt file
def write_to_label_file(name, data):
    with open(name,"w") as o:
        for row in data:
            line = str(row) + "\n"
            o.write(line)

# write image txt file
def write_to_image_file(name, data):
    with open(name,"w") as o:
        for row in data:
            line = ""
            for element in row:
                line += str(element) + " "
            line += "\n"
            o.write(line)

def fold_split():
    train_image = read_train_image("train_image.txt")
    train_label = read_train_label("train_label.txt")
    test_image = read_train_image("test_image.txt")
    test_label = read_train_label("test_label.txt")
    total_image = np.concatenate((train_image, test_image), axis=0)
    total_label = np.concatenate((train_label, test_label), axis=0)
    total_size = total_image.shape[0]
    ans = np.random.permutation(total_size)
    slice = total_size // FOLD
    for i in range(FOLD):
        s = i*slice
        e = (i+1)*slice
        temp_test_image = total_image[ans[s:e]]
        temp_test_label = total_label[ans[s:e]]
        if i == 0:
            temp_train_image = total_image[ans[e:]]
            temp_train_label = total_label[ans[e:]]
        elif i == FOLD-1:
            temp_train_image = total_image[ans[0:s]]
            temp_train_label = total_label[ans[0:s]]
        else:
            temp_train_image = np.concatenate((total_image[ans[0:s]], total_image[ans[e:]]), axis=0)
            temp_train_label = np.concatenate((total_label[ans[0:s]], total_label[ans[e:]]), axis=0)
        head = "fold" + str(i+1)
        write_to_image_file("fold/%s_train_image.txt" %(head), temp_train_image)
        write_to_image_file("fold/%s_test_image.txt" %(head), temp_test_image)
        write_to_label_file("fold/%s_train_label.txt" %(head), temp_train_label)
        write_to_label_file("fold/%s_test_label.txt" %(head), temp_test_label)

if __name__ == "__main__":
    fold_split()