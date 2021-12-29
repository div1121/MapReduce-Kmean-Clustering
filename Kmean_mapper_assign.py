#/usr/bin/env python

import sys
import numpy as np

# Ground truth number
K = 10

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
for line in sys.stdin:
    key, value = line.split(":")
    feature = value.split()
    feature = list(map(float, feature))
    row = np.array(feature)
    min_distance = sys.float_info.max
    min_index = None
    for center_index in range(K):
        dis = distance(row, centroids[center_index])
        if dis < min_distance:
            min_distance = dis
            min_index = center_index
    print("%s\t%s" %(key, min_index))



