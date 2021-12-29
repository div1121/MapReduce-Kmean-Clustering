import sys
import numpy as np
import random
from random import randrange

# change the seed for random centroids points
SEED = 51

random.seed(SEED)

# Ground truth number
K = 10

# Dimension
d = 28*28

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

# random one point and then furthest distance approach
def initialize_centroid(data):
    first = randrange(data.shape[0])
    centroids = [data[first]]
    for t in range(K-1):
        max_distance = 0
        max_row = None
        for row in data:
            min_distance = sys.float_info.max
            for center in centroids:
                dis = distance(row, center)
                min_distance = min(dis, min_distance)
            if min_distance > max_distance:
                max_row = row
                max_distance = min_distance
        centroids.append(max_row)
    return centroids

# parse centroids format
def parse_string(center_sum):
    save = ""
    for i in range(d):
        if i == d-1:
            save += str(center_sum[i])
        else:
            save += str(center_sum[i]) + ","
    return save

# write centroids to centroids file
def write_file(name, centroids):
    with open(name,"w") as f:
        for i in range(len(centroids)):
            value = parse_string(centroids[i])
            ans = str(i) + "\t" + value + "\n"
            f.write(ans)

# main process
def random_centroids():
    dataset = read_train_image("train_image.txt")
    centroids = initialize_centroid(dataset)
    write_file("centroids",centroids)

if __name__ == "__main__":
    random_centroids()