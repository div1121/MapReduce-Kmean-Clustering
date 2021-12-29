import sys
import numpy as np
import random
from random import randrange

# Ground truth
K = 10

# evaluate training data result
def evaluate_train_result(label, dataset_group):
    groups = [[0 for i in range(K)] for j in range(K)]
    ans_group = []
    for i in range(len(label)):
        x = dataset_group[i]
        y = label[i]
        groups[x][y] = groups[x][y] + 1
    sum_total = 0
    sum_correct = 0
    for i in range(K):
        group = groups[i]
        a = np.array(group)
        idx = np.argmax(a)
        total = sum(group)
        correct = max(group)
        acc = float(correct) / float(total)
        sum_total = sum_total + total
        sum_correct = sum_correct + correct
        ans_group.append(idx)
        print("Number of Train images in cluster %s: %s, Major Label: %s, Number of Corrected Images: %s, Classification Accuracy: %s" %(i, total, idx, correct, acc))
    total_acc = float(sum_correct) / float(sum_total)
    print("Number of Train images: %s, Number of Corrected Images: %s, Classification Accuracy: %s" %(sum_total, sum_correct, total_acc))
    return ans_group

# predicted label of training data
def read_predicted_label(path):
    dataset_group = []
    with open(path,"r") as f:
        for line in f.readlines():
            key, value = line.split("\t")
            value = int(value)
            dataset_group.append(value)
    return dataset_group

# ground truth label
def read_train_label(path):
    label = []
    with open(path,"r") as f:
        for line in f.readlines():
            label.append(int(line))
    return np.array(label) 

# read centroids
def read_centroids(path):
    centroids = []
    with open(path,"r") as f:
        for line in f.readlines():
            if "Error" not in line:
                key, value = line.split("\t")
                feature = value.split(",")
                feature = list(map(float, feature))
                center = np.array(feature)
                centroids.append(center)
    return centroids

# read image
def read_train_image(path):
    dataset = []
    with open(path,"r") as f:
        for line in f.readlines():
            feature = line.split()
            feature = list(map(float, feature))
            dataset.append(feature)
    return np.array(dataset)

# square of eculidean distance
def distance(a,b):
    dif = a - b
    return np.sum(np.square(dif))

# assign group to dataset
def assign_group(data, centroids):
    dataset_group = []
    for row in data:
        min_distance = sys.float_info.max
        min_index = None
        for center_index in range(K):
            dis = distance(row, centroids[center_index])
            if dis < min_distance:
                min_distance = dis
                min_index = center_index
        dataset_group.append(min_index)
    return dataset_group

# evaluate testing data result
def evaluate_test_result(label, dataset_group, ans_group):
    sum_total = 0
    sum_correct = 0
    for i in range(len(label)):
        x = dataset_group[i]
        y = label[i]
        if ans_group[x] == y:
            sum_correct += 1
        sum_total += 1
    acc = float(sum_correct) / float(sum_total)
    print("Test Classification Accuracy: %s" %(acc))

def evaluate():
    # Training set evaluation
    dataset_group = read_predicted_label("ordered_labels")
    label = read_train_label("fold/fold5_train_label.txt")
    ans_group = evaluate_train_result(label, dataset_group)
    # Testing set evaluation
    test_dataset = read_train_image("fold/fold5_test_image.txt")
    centroids = read_centroids("centroids")
    test_predict_group = assign_group(test_dataset, centroids)
    test_label = read_train_label("fold/fold5_test_label.txt")
    evaluate_test_result(test_label, test_predict_group, ans_group)

if __name__ == "__main__":
    evaluate()