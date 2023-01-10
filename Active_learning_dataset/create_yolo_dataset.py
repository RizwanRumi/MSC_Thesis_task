import argparse
import fnmatch
import os
import random
import shutil
from pathlib import Path

ROOT_DIR = "./Training_pool_dataset/"
pool_test_directory = ROOT_DIR + "test/"
ITERATION_ROOT_DIR = "./Iteration_dataset/"
iterative_train_dir = ITERATION_ROOT_DIR + "train/"
iterative_val_dir = ITERATION_ROOT_DIR + "valid/"

def create_sub_folders(destination_dir):
    destination_img_dir = destination_dir + "images/"
    destination_label_dir = destination_dir + "labels/"
    os.makedirs(destination_img_dir, exist_ok=True)
    os.makedirs(destination_label_dir, exist_ok=True)

def main():
    print("This is me")
    iteration = 25
    dest_folder_list = []
    final_dataset_root_dir = str("ACTIVE_LEARNING_DATASETS/iteration_{}_dataset").format(iteration)
    train_dir = final_dataset_root_dir + "/train/"
    dest_folder_list.append(train_dir)
    valid_dir = final_dataset_root_dir + "/valid/"
    dest_folder_list.append(valid_dir)
    test_dir = final_dataset_root_dir + "/test/"
    dest_folder_list.append(test_dir)

    tr_vl_ts = 0
    create_sub_folders(train_dir)
    create_sub_folders(valid_dir)
    create_sub_folders(test_dir)

    source_folder_list = []
    source_train_broken = iterative_train_dir + "Broken/"
    source_train_healthy = iterative_train_dir + "Healthy/"
    source_folder_list.append(source_train_broken)
    source_folder_list.append(source_train_healthy)

    source_val_broken = iterative_val_dir + "Broken/"
    source_val_healthy = iterative_val_dir + "Healthy/"
    source_folder_list.append(source_val_broken)
    source_folder_list.append(source_val_healthy)

    source_test_broken = pool_test_directory + "Broken/"
    source_test_healthy = pool_test_directory + "Healthy/"
    source_folder_list.append(source_test_broken)
    source_folder_list.append(source_test_healthy)

    flag1 = 0
    flag2 = 2
    for i in range(len(dest_folder_list)):
        for j in range(flag1, flag2):
            print("-------------")
            print("src = {} , dst = {} ".format(source_folder_list[j], dest_folder_list[i]))
            shutil.copytree(source_folder_list[j], dest_folder_list[i], symlinks=False, ignore=None,
                            copy_function=shutil.copy, ignore_dangling_symlinks=False, dirs_exist_ok=True)
            flag1 += 1
        flag2 += 2

    print("Transfer data successfully")
    #

    return 0

if __name__ == "__main__":
    main()
