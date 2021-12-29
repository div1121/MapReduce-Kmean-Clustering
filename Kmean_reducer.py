#/usr/bin/env python

import sys
import numpy as np

# Dimension
d = 28*28

# keep track
prev_key = None
prev_center = np.zeros((d), dtype=float)
prev_count = 0
errors = 0

# format centroid
def parse_string(center_sum):
    save = ""
    for i in range(d):
        if i == d-1:
            save += str(center_sum[i])
        else:
            save += str(center_sum[i]) + ","
    return save

# centroid update
for line in sys.stdin:
    key, value = line.split("\t")
    center_str, count, error = value.split("|")
    center_list = center_str.split(",")
    center = list(map(float, center_list))
    center = np.array(center)
    count = int(count)
    error = float(error)

    if key == prev_key:
        prev_center += center
        prev_count += count
        errors += error
    else:
        if prev_key:
            real_center = prev_center / prev_count
            value = parse_string(real_center)
            print("%s\t%s" %(prev_key, value))
        prev_key = key
        prev_center = center
        prev_count = count
        errors += error

if prev_key:
    real_center = prev_center / prev_count
    value = parse_string(real_center)
    print("%s\t%s" %(prev_key, value))

print("%s\t%s" %("Error",errors))
