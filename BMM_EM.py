import sys
import numpy as np
import random
from random import randrange
import math

SEED = 10

random.seed(SEED)

# Ground truth number
K = 10

# Dimension
D = 28*28

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

# random initialize parameters
def random_initialize_parameter():
    bnn = np.random.rand(K,D)
    weighting = np.random.rand(K)
    weighting = weighting / np.sum(weighting)
    return bnn, weighting

# data assignment probability
def assignment_data_probability(data_row, bnn_row, scale):
    fake_bnn = np.copy(bnn_row)
    fake_bnn[data_row==0.0] = 1 - fake_bnn[data_row==0.0]
    fake_bnn = fake_bnn / scale
    return np.prod(fake_bnn)

def counter_underflow_computation(bnn):
    # trick for dealing underflow
    # mean for probability
    a = np.mean(bnn)
    fake_bnn = 1 - bnn
    b = np.mean(fake_bnn)
    return 0.4

# data assignment
def data_assignment(data, bnn, weighting):
    precompute = counter_underflow_computation(bnn)
    # print(precompute)
    bnn_sum = [np.zeros(data.shape[1], dtype=float)] * K
    weighting_sum = [0.0] * K
    total_sum = [0.0] * K
    gains = 0.0
    for row in data:
        row_kcluster_num = []
        for center_index in range(K):
            score = weighting[center_index] * assignment_data_probability(row, bnn[center_index], precompute)
            #print(score)
            row_kcluster_num.append(score)
        sum_row_kcluster_num = sum(row_kcluster_num)
        gains += np.log(sum_row_kcluster_num)
        # print(errors)
        for center_index in range(K):
            real_score = row_kcluster_num[center_index] / sum_row_kcluster_num
            scale_row = real_score * row
            bnn_sum[center_index] = bnn_sum[center_index] + scale_row
            weighting_sum[center_index] = weighting_sum[center_index] + real_score
            total_sum[center_index] = total_sum[center_index] + 1
    # print(gains)
    return bnn_sum, weighting_sum, total_sum, [gains for i in range(K)]

# update parameters
def update_parameters(bnn_sum, weighting_sum, total_sum):
    # print(cluster_count)
    new_bnn_sum = []
    new_weighting_sum = []
    for i in range(K):
        new_bnn = bnn_sum[i] / weighting_sum[i]
        new_weighting = weighting_sum[i] / total_sum[i]
        new_bnn_sum.append(new_bnn)
        new_weighting_sum.append(new_weighting)
    return np.array(new_bnn_sum), np.array(new_weighting_sum)

# assign group to dataset
def assign_group(data, bnn, weighting):
    precompute = counter_underflow_computation(bnn)
    # print(precompute)
    dataset_group = []
    for row in data:
        max_score = 0.0
        max_index = 0
        for center_index in range(K):
            score = weighting[center_index] * assignment_data_probability(row, bnn[center_index], precompute)
            if max_score < score:
                max_score = score
                max_index = center_index
        dataset_group.append(max_index)
    return dataset_group

# evaluate train result
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

# evaluate test result
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

# BMM-EM clustering
def BMM_EM():
    dataset = read_train_image("train_image.txt")
    print("Finish Read image data")
    dataset[dataset<127.5] = 0.0
    dataset[dataset>=127.5] = 1.0
    bnn, weighting = random_initialize_parameter()
    print("Finish initialize parameter")
    # print((bnn>0).all())
    # print(weighting)
    prev_gain = -sys.float_info.max
    cur_gain = None
    count = 0
    while (cur_gain is None or cur_gain - prev_gain >= 0.1):
        if cur_gain is not None:
            prev_gain = cur_gain
        bnn_sum, weighting_sum, total_sum, gains = data_assignment(dataset, bnn, weighting)
        cur_gain = sum(gains) / K
        bnn, weighting = update_parameters(bnn_sum, weighting_sum, total_sum)
        count += 1
        print ("Epoch %s Gain %s" %(count, cur_gain))
    print("Result:")
    # Training set evaluation
    label = read_train_label("train_label.txt")
    data_groups = assign_group(dataset, bnn, weighting)
    ans_group = evaluate_train_result(label, data_groups)
    # Testing set evaluation
    test_dataset = read_train_image("test_image.txt")
    test_predict_group = assign_group(test_dataset, bnn, weighting)
    test_label = read_train_label("test_label.txt")
    evaluate_test_result(test_label, test_predict_group, ans_group)
    
if __name__ == "__main__":
    BMM_EM()