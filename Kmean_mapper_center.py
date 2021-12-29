#/usr/bin/env python

import sys
import numpy as np

# Ground truth number
K = 10

# Dimension
d = 28*28

# read centroids
centroids = []
with open("centroids","r") as f:
    for line in f.readlines():
        if "Error" not in line:
            key, value = line.split("\t")
            feature = value.split(",")
            feature = list(map(float, feature))
            center = np.array(feature)
            centroids.append(center)

# square of eculidean distance
def distance(a,b):
    dif = a - b
    return np.sum(np.square(dif))

# data assignment
cluster_sum = [np.zeros(d, dtype=float)] * K
cluster_count = [0] * K
cluster_error = [0.0] * K

for line in sys.stdin:
    feature = line.split()
    feature = list(map(float, feature))
    row = np.array(feature)
    min_distance = sys.float_info.max
    min_index = None
    for center_index in range(K):
        dis = distance(row, centroids[center_index])
        if dis < min_distance:
            min_distance = dis
            min_index = center_index
    cluster_sum[min_index] = cluster_sum[min_index] + row
    cluster_count[min_index] = cluster_count[min_index] + 1
    cluster_error[min_index] = cluster_error[min_index] + min_distance

# output key and value
def parse_string(center_sum):
    save = ""
    for i in range(d):
        if i == d-1:
            save += str(center_sum[i])
        else:
            save += str(center_sum[i]) + ","
    return save

for i in range(K):
    center_sum = cluster_sum[i]
    first = parse_string(center_sum)
    count = cluster_count[i]
    error = cluster_error[i]
    value = first + "|" + str(count) + "|" + str(error)
    print("%s\t%s" %(i, value))



