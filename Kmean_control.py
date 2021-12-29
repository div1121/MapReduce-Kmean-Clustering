import os
import sys
import subprocess

# put training data
os.system("hdfs dfs -put train_image.txt train_image.txt")  

# read file for updating error
def read_file_error(file):
    with open(file,"r") as f:
        for line in f.readlines():
            if "Error" in line:
                key, value = line.split("\t")
                return float(value)

# K-mean looping until convergence
prev_error = sys.float_info.max
cur_error = None
count = 0
while (cur_error is None or prev_error - cur_error >= 0.1):
    # one iteration of K-mean
    cmd = 'hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar -file centroids -file Kmean_mapper.py -mapper "python3 Kmean_mapper.py" -file Kmean_reducer.py -reducer "python3 Kmean_reducer.py" -input train_image.txt -output output-%d' %(count)
    os.system(cmd)
    if cur_error is not None:
        prev_error = cur_error
    # output centroid
    output = "hdfs dfs -cat output-%d/* > centroids-%d" %(count, count+1)
    os.system(output)
    count += 1
    # update the loss
    file = "centroids-%d" %(count)
    cur_error = read_file_error(file)
    print ("Epoch %s Loss %s" %(count, cur_error))
    # update centroids
    os.system("rm centroids")
    os.system("mv centroids-%d centroids" %(count))
    if count >= 100 or cur_error is None:
        break

# output centroid information
os.system('hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar -file centroids -file Kmean_mapper_center.py -mapper "python3 Kmean_mapper_center.py" -file Kmean_reducer_center.py -reducer "python3 Kmean_reducer_center.py" -input train_image.txt -output output-final')
os.system("hdfs dfs -cat output-final/* > clusters")

# output label of data (where it is label as 0,1,2,..)
os.system("hdfs dfs -put special_train_image.txt special_train_image.txt")
os.system('hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar -file centroids -file Kmean_mapper_assign.py -mapper "python3 Kmean_mapper_assign.py" -file Kmean_reducer_assign.py -reducer "python3 Kmean_reducer_assign.py" -input special_train_image.txt -output output-label')
os.system("hdfs dfs -cat output-label/* > labels")
os.system("sort -n labels > ordered_labels")
