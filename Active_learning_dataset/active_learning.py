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

ROOT_DIR = "./new_dataset/"
temp_img_directory = ROOT_DIR + "temp_images/"
temp_text_directory = ROOT_DIR + "temp_labels/"


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


def get_pooling_data(number_of_pooling_data, img_dir, label_dir):
    # Pool data by selecting the equal number of healthy and broken images with their labels in temporary folders
    os.makedirs(temp_img_directory, exist_ok=True)
    os.makedirs(temp_text_directory, exist_ok=True)
    images = []
    labels = []

    # check the even percentage to pool balanced data
    if number_of_pooling_data % 2 != 0:
        number_of_pooling_data = number_of_pooling_data - 1

    # divide the percentage of an equal number for healthy and broken images
    select_healthy = int(number_of_pooling_data / 2)
    select_broken = int(number_of_pooling_data - select_healthy)

    # get equal number of random healthy and broken data
    healthy_images = random.sample([f for f in os.listdir(img_dir) if 'healthy' in f], select_healthy)
    images.extend(healthy_images)

    broken_images = random.sample([f for f in os.listdir(img_dir) if 'broken' in f], select_broken)
    images.extend(broken_images)

    # get name list from images to check the labels file
    names = [os.path.splitext(filename)[0] for filename in images]

    # get labels
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
        for name in images:
            img_source_path = os.path.join(img_dir, name)
            shutil.move(img_source_path, temp_img_directory)

        for file in names:
            txt = file + '.txt'
            labels.append(txt)
            txt_source_path = os.path.join(label_dir, txt)
            shutil.move(txt_source_path, temp_text_directory)

        print("-------------------------")
        print("image file moved: ", len(images))
        print("text file moved: ", len(labels))
    else:
        raise Exception("Text files are missing..")
    # return images, labels
    return names


def split_data(selected_files):
    # data split between train, validation and test
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
        train_img_path = os.path.join(img_dir, image)
        shutil.move(train_img_path, ROOT_DIR + "train/images/")
        train_txt_path = os.path.join(label_dir, label)
        shutil.move(train_txt_path, ROOT_DIR + "train/labels/")

    for item in val_files:
        image = item + '.jpg'
        label = item + '.txt'
        val_img_path = os.path.join(img_dir, image)
        shutil.move(val_img_path, ROOT_DIR + "valid/images/")
        val_txt_path = os.path.join(label_dir, label)
        shutil.move(val_txt_path, ROOT_DIR + "valid/labels/")

    for item in test_files:
        image = item + '.jpg'
        label = item + '.txt'
        test_img_path = os.path.join(img_dir, image)
        shutil.move(test_img_path, ROOT_DIR + "test/images/")
        test_txt_path = os.path.join(label_dir, label)
        shutil.move(test_txt_path, ROOT_DIR + "test/labels/")


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
        default="./dataset_1"
    )
    parser.add_argument(
        "-I",
        "--iteration",
        help="Number of iteration to pick selected number of random data from main dataset",
        type=int,
        default=1
    )

    try:
        args = parser.parse_args()
        assert os.path.isdir(args.inputDir), "The directory is not found. Please check the directory."

        ITERATION = args.iteration
        assert ITERATION > 0, "Only positive numbers are allowed"

        print("********** Iteration: {} **********".format(ITERATION))
        print("-------------------------")
        # for images and labels folder
        directory = args.inputDir
        folders = os.listdir(directory)
        images_directory = Path(directory) / folders[0]
        labels_directory = Path(directory) / folders[1]

        total_images = get_images_and_labels(images_directory, labels_directory)
        print("Images and Labels (both) are : ", total_images)

        if ITERATION == 1:
            selected_ratio = 0.15
        else:
            selected_ratio = 0.05

        # Create new dataset directory
        os.makedirs(ROOT_DIR, exist_ok=True)
        number_of_pooling_data = int(total_images * selected_ratio)
        print("Data Select for Pooling: ", number_of_pooling_data)

        selected_files = get_pooling_data(number_of_pooling_data, images_directory, labels_directory)
        split_data(selected_files)

        print("-------------------------")
        print("Split Successful. Current details:: ")
        print('Training: ', get_images_and_labels(ROOT_DIR + 'train/images', ROOT_DIR + 'train/labels'))
        print('Validation: ', get_images_and_labels(ROOT_DIR + 'valid/images', ROOT_DIR + 'valid/labels'))
        print('Testing: ', get_images_and_labels(ROOT_DIR + 'test/images', ROOT_DIR + 'test/labels'))
        print("-------------------------")

        if len(os.listdir(temp_img_directory)) == 0 and len(os.listdir(temp_text_directory)) == 0:
            os.rmdir(temp_img_directory)
            os.rmdir(temp_text_directory)
        else:
            raise Exception('Temporary Folders could not be deleted.')

    except AssertionError as msg:
        print(msg)


if __name__ == "__main__":
    main()
