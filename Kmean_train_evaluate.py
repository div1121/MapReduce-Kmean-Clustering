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

def evaluate():
    dataset_group = read_predicted_label("ordered_labels")
    label = read_train_label("fold/fold1_train_label.txt")
    ans_group = evaluate_train_result(label, dataset_group)

if __name__ == "__main__":
    evaluate()