import numpy as np
import struct

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

# train image file reading
with open('train-images.idx3-ubyte', 'rb') as f:
    magic, size = struct.unpack('>II', f.read(8))
    nrows, ncols = struct.unpack('>II', f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8)).newbyteorder(">")
    data = data.reshape((size,nrows*ncols))
    write_to_image_file("train_image.txt", data)

# train label file reading
with open('train-labels.idx1-ubyte', 'rb') as i:
    magic, size = struct.unpack('>II', i.read(8))
    data_1 = np.fromfile(i, dtype=np.dtype(np.uint8)).newbyteorder(">")
    write_to_label_file("train_label.txt", data_1)

# test image file reading
with open('t10k-images.idx3-ubyte', 'rb') as f:
    magic, size = struct.unpack('>II', f.read(8))
    nrows, ncols = struct.unpack('>II', f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8)).newbyteorder(">")
    data = data.reshape((size,nrows*ncols))
    write_to_image_file("test_image.txt", data)

# test label file reading
with open('t10k-labels.idx1-ubyte', 'rb') as i:
    magic, size = struct.unpack('>II', i.read(8))
    data_1 = np.fromfile(i, dtype=np.dtype(np.uint8)).newbyteorder(">")
    write_to_label_file("test_label.txt", data_1)