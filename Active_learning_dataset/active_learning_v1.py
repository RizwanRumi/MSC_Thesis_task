"""
Use:
python active_learning.py -dir ./dataset_1 -I 1
python active_learning.py -dir ./dataset_1 -I 2
    or
python active_learning.py --inputDir ./dataset_1 --iteration 1
python active_learning.py --inputDir ./dataset_1 --iteration 2
"""

import argparse
import fnmatch
import os
import random
import shutil
from pathlib import Path

import numpy as np

ROOT_DIR = "./Training_pool_dataset/"
pool_train_directory = ROOT_DIR + "train/"
pool_test_directory = ROOT_DIR + "test/"

ITERATION_ROOT_DIR = "./Iteration_dataset/"
iterative_train_dir = ITERATION_ROOT_DIR + "train/"
iterative_val_dir = ITERATION_ROOT_DIR + "val/"

temp_img_directory = ROOT_DIR + "temp_images/"
temp_text_directory = ROOT_DIR + "temp_labels/"

train_data_ratio = 0.80
test_data_ratio = 0.20

def get_images_and_labels(img_path, label_path):
    img_count = 0
    label_count = 0

    for r, d, files in os.walk(img_path):
        img_count += len(fnmatch.filter(os.listdir(r), '*.jpg'))

    for r, d, files in os.walk(label_path):
        label_count += len(fnmatch.filter(os.listdir(r), '*.txt'))

    if img_count != label_count:
        raise Exception('images and labels are not equaled in Dataset')
    else:
        return img_count


def split_ratio(selected_for_training, img_dir, label_dir):
    os.makedirs(temp_img_directory, exist_ok=True)
    os.makedirs(temp_text_directory, exist_ok=True)
    images = []
    labels = []

    if selected_for_training % 2 != 0:
        selected_for_training = selected_for_training + 1

    select_healthy = int(selected_for_training / 2)
    select_broken = int(selected_for_training - select_healthy)

    healthy_images = random.sample([f for f in os.listdir(img_dir) if 'healthy' in f], select_healthy)
    images.extend(healthy_images)

    broken_images = random.sample([f for f in os.listdir(img_dir) if 'broken' in f], select_broken)
    images.extend(broken_images)

    names = [os.path.splitext(filename)[0] for filename in images]
    text_files = [f for f in os.listdir(label_dir)]

    # Check missing labelled text file from labels folder.
    missing_file = []

    for filename in names:
        chk = any(file for file in text_files if filename in file)

        if chk:
            continue
        else:
            missing_file.append(filename)

    if len(missing_file) == 0:
        for fname in images:
            srcpath = os.path.join(img_dir, fname)
            shutil.move(srcpath, temp_img_directory)

        for file in names:
            txt = file + '.txt'
            labels.append(txt)
            srcpath = os.path.join(label_dir, txt)
            shutil.move(srcpath, temp_text_directory)

        print("-------------------------")
        print("image file moved: ", len(images))
        print("text file moved: ", len(labels))
    else:
        raise Exception("Text files are missing..")
    # return images, labels
    return names

def train_test_split(selected_files):
    os.makedirs(ROOT_DIR + "train/images/", exist_ok=True)
    os.makedirs(ROOT_DIR + "train/labels/", exist_ok=True)
    os.makedirs(ROOT_DIR + "valid/images/", exist_ok=True)
    os.makedirs(ROOT_DIR + "valid/labels/", exist_ok=True)
    os.makedirs(ROOT_DIR + "test/images/", exist_ok=True)
    os.makedirs(ROOT_DIR + "test/labels/", exist_ok=True)

    # Creating partitions of the data after shuffling
    val_ratio = 0.20
    test_ratio = 0.10
    np.random.shuffle(selected_files)

    train_files, val_files, test_files = np.split(np.array(selected_files),
                                                  [int(len(selected_files) * (1 - (val_ratio + test_ratio))),
                                                   int(len(selected_files) * (1 - test_ratio))])
    print("-------------------------")
    print('Training: ', len(train_files))
    print('Validation: ', len(val_files))
    print('Testing: ', len(test_files))

    img_dir = temp_img_directory
    label_dir = temp_text_directory

    for item in train_files:
        image = item + '.jpg'
        label = item + '.txt'
        imgpath = os.path.join(img_dir, image)
        shutil.move(imgpath, ROOT_DIR + "train/images/")
        txtpath = os.path.join(label_dir, label)
        shutil.move(txtpath, ROOT_DIR + "train/labels/")

    for item in val_files:
        image = item + '.jpg'
        label = item + '.txt'
        imgpath = os.path.join(img_dir, image)
        shutil.move(imgpath, ROOT_DIR + "valid/images/")
        txtpath = os.path.join(label_dir, label)
        shutil.move(txtpath, ROOT_DIR + "valid/labels/")

    for item in test_files:
        image = item + '.jpg'
        label = item + '.txt'
        imgpath = os.path.join(img_dir, image)
        shutil.move(imgpath, ROOT_DIR + "test/images/")
        txtpath = os.path.join(label_dir, label)
        shutil.move(txtpath, ROOT_DIR + "test/labels/")

def write_selected_files(directory, file_name, file_list):
    # create file name with directory
    file_path = directory + file_name + '.txt'
    with open(file_path, 'w') as f:
        for name in file_list:
            f.write("%s\n" % name)

def split_data(number_of_data_selection, source_img_dir, source_label_dir, destination_img_dir, destination_label_dir):
    images = []
    labels = []

    select_healthy = int(number_of_data_selection / 2)
    select_broken = int(number_of_data_selection - select_healthy)

    # select random data
    healthy_images = random.sample([f for f in os.listdir(source_img_dir) if 'healthy' in f], select_healthy)
    images.extend(healthy_images)

    broken_images = random.sample([f for f in os.listdir(source_img_dir) if 'broken' in f], select_broken)
    images.extend(broken_images)

    names = [os.path.splitext(filename)[0] for filename in images]
    text_files = [f for f in os.listdir(source_label_dir)]

    # Check missing labelled text file from labels folder.
    missing_file = []

    for filename in names:
        chk = any(file for file in text_files if filename in file)

        if chk:
            continue
        else:
            missing_file.append(filename)

    if len(missing_file) == 0:
        for name in images:
            source_img_path = os.path.join(source_img_dir, name)
            shutil.move(source_img_path, destination_img_dir)

        for file in names:
            txt = file + '.txt'
            labels.append(txt)
            source_label_path = os.path.join(source_label_dir, txt)
            shutil.move(source_label_path, destination_label_dir)

        print("-------------------------")
        print(str(destination_img_dir) + ":")
        print("image file moved: ", len(images))
        print(str(destination_label_dir) + ":")
        print("text file moved: ", len(labels))
    else:
        raise Exception("Text files are missing..")
    # return images, labels
    return names

def create_sub_folders(destination_dir):
    destination_img_dir = destination_dir + "images/"
    destination_label_dir = destination_dir + "labels/"
    os.makedirs(destination_img_dir, exist_ok=True)
    os.makedirs(destination_label_dir, exist_ok=True)

def get_image_and_label_folder(directory):
    folders = os.listdir(directory)
    img_dir = Path(directory) / folders[0]
    label_dir = Path(directory) / folders[1]
    return img_dir, label_dir

def create_pooling_data(iteration, total_images, img_dir, label_dir):
    # data split between train (80%) and test (20%) folders
    if iteration == 0:
        if os.path.exists(pool_train_directory) and os.path.exists(pool_test_directory):
            raise Exception('Data Pooling already done, please try to create iterative dataset')
        else:
            train_directory = pool_train_directory
            test_directory = pool_test_directory
            create_sub_folders(train_directory)
            create_sub_folders(test_directory)
            save_train_name = "Pool_train_data_names"
            save_test_name = "Pool_test_data_names"

    else:
        train_directory = iterative_train_dir
        test_directory = iterative_val_dir
        create_sub_folders(train_directory)
        create_sub_folders(test_directory)
        save_train_name = "iteration_train_dataset_" + str(iteration)
        save_test_name = "iteration_val_dataset_" + str(iteration)

    number_of_train_data = int(total_images * train_data_ratio)

    if number_of_train_data % 2 != 0:
        number_of_train_data = number_of_train_data + 1

    #number_of_test_data = total_images - number_of_train_data
    number_of_test_data = int(total_images * test_data_ratio)

    source_img_dir = img_dir
    source_label_dir = label_dir

    """
    Train and test data will be split from the main dataset folders
    """

    # Create train data
    destination_img_dir, destination_label_dir = get_image_and_label_folder(train_directory)
    # split train data from the source folder
    filenames = split_data(number_of_train_data, source_img_dir, source_label_dir,
                           destination_img_dir, destination_label_dir)
    # save selected training data
    write_selected_files(train_directory, save_train_name, filenames)

    # Create test / val data
    destination_img_dir, destination_label_dir = get_image_and_label_folder(test_directory)
    filenames = split_data(number_of_test_data, source_img_dir, source_label_dir,
                           destination_img_dir, destination_label_dir)
    # save selected test / val data
    write_selected_files(test_directory, save_test_name, filenames)

    #return number_of_train_data

def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="Apply active learning and split the dataset"
    )
    parser.add_argument(
        "-dir",
        "--inputDir",
        help="Path to the folder where the input images and labels are stored",
        type=str,
        default="./test_dataset"
    )
    parser.add_argument(
        "-I",
        "--iteration",
        help="Number of iteration to pick selected number of random data from main dataset",
        type=int,
        default=0
    )

    try:
        args = parser.parse_args()
        assert os.path.isdir(args.inputDir), "The directory is not found. Please check the directory."

        ITERATION = args.iteration
        # assert ITERATION > 0, "Only positive numbers are allowed"

        print("********** Iteration: {} **********".format(ITERATION))
        print("-------------------------")
        directory = args.inputDir

        if ITERATION == 0:

            folders = os.listdir(directory)
            images_directory = Path(directory) / folders[0]
            labels_directory = Path(directory) / folders[1]
            
            # Check the equal number of images and labels from dataset folder
            total_images = get_images_and_labels(images_directory, labels_directory)
            print("Images and Labels (both) are : ", total_images)

            # Create Training Pool
            os.makedirs(ROOT_DIR, exist_ok=True)

            create_pooling_data(ITERATION, total_images, images_directory, labels_directory)
        else:
            if ITERATION == 1:
                selected_ratio = 0.15
            else:
                selected_ratio = 0.05

            # Create iterative dataset directory
            os.makedirs(ITERATION_ROOT_DIR, exist_ok=True)

            pool_train_folders = os.listdir(pool_train_directory)
            pool_images_directory = Path(directory) / pool_train_folders[0]
            pool_labels_directory = Path(directory) / pool_train_folders[1]

            total_selected_training_data = get_images_and_labels(pool_images_directory, pool_labels_directory)
            print("Selected for Training data: ", total_selected_training_data)

            create_pooling_data(ITERATION, total_selected_training_data, pool_images_directory, pool_labels_directory)

            print("-------------------------")
            print("Split Successful. Current details:: ")
            print('Training: ', get_images_and_labels(iterative_train_dir + 'images/', iterative_train_dir + 'labels/'))
            print('Validation: ', get_images_and_labels(iterative_val_dir + 'images', iterative_val_dir + 'labels/'))
            print('Testing: ', get_images_and_labels(ROOT_DIR + 'test/images', ROOT_DIR + 'test/labels'))
            print("-------------------------")

    except AssertionError as msg:
        print(msg)


if __name__ == "__main__":
    main()
