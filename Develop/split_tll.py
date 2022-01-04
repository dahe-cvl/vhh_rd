left_path = "/caa/Homes01/fjogl/vhh_rd/left"
right_path = "/caa/Homes01/fjogl/vhh_rd/left"

target = "/caa/Homes01/fjogl/vhh_rd/datasets"
sets = ["train", "test", "val"]

test_val_percentage = 0.1

import os, glob
import numpy as np
import shutil

def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def move(nr, name):
    mkdir(os.path.join(target, name))
    mkdir(os.path.join(target, name, "left"))
    mkdir(os.path.join(target, name, "right"))


    files = [os.path.split(p)[-1] for p in glob.glob(os.path.join(left_path, "*.jpg"))]
    selected_files = np.random.choice(files, size=nr, replace=False)
    for file in selected_files:
        shutil.copy2(os.path.join(left_path, file), os.path.join(target, name, "left", file))
        shutil.copy2(os.path.join(right_path, file), os.path.join(target, name, "right", file))

mkdir(target)
files = [os.path.split(p)[-1] for p in glob.glob(os.path.join(left_path, "*.jpg"))]

len_test_val = int(len(files)*test_val_percentage)
len_train = len(files) - 2*len_test_val

move(len_test_val, "test")
move(len_test_val, "val")
move(len_train, "train")

