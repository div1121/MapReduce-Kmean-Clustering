def read_train_image(input_path, output_path):
    count = 0
    with open(input_path,"r") as i:
        with open(output_path, "w") as o:
            for line in i.readlines():
                output_line = str(count) + ":" + line
                o.write(output_line)
                count += 1

if __name__ == "__main__":
    read_train_image("train_image.txt","special_train_image.txt")