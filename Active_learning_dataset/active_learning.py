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


def split_ratio(selected_for_training: object, img_dir, label_dir):
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


def main():
    global INITIAL_SPLIT

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
        selected_for_training = int(total_images * selected_ratio)
        print("Selected for Training: ", selected_for_training)

        selected_files = split_ratio(selected_for_training, images_directory, labels_directory)
        train_test_split(selected_files)

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
