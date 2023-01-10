"""
Use:
Training Pool:
---------------------
python active_learning_v2.py -dir ./test_dataset -I 0
    or
python active_learning_v2.py --inputDir ./test_dataset --iteration 0

query selection:
----------------------
python active_learning_v2.py -dir ./Training_pool_dataset -I 1
    or
python active_learning_v2.py --inputDir ./Training_pool_dataset --iteration 1
"""

import argparse
import fnmatch
import os
import random
import shutil
from pathlib import Path

ROOT_DIR = "./Training_pool_dataset/"
pool_train_directory = ROOT_DIR + "train/"
pool_test_directory = ROOT_DIR + "test/"

QUERY_SELECT_DIR = "./Query_selection_dataset/"

ITERATION_ROOT_DIR = "./Iteration_dataset/"
iterative_train_dir = ITERATION_ROOT_DIR + "train/"
iterative_val_dir = ITERATION_ROOT_DIR + "valid/"

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


def write_selected_files(directory, file_name, file_list):
    # create file name with directory
    file_path = directory + file_name + '.txt'
    with open(file_path, 'w') as f:
        for name in file_list:
            f.write("%s\n" % name)


def create_sub_folders(destination_dir):
    destination_img_dir = destination_dir + "images/"
    destination_label_dir = destination_dir + "labels/"
    os.makedirs(destination_img_dir, exist_ok=True)
    os.makedirs(destination_label_dir, exist_ok=True)


def get_folder_list(directory):
    folders = os.listdir(directory)
    directory_1 = Path(directory) / folders[0]
    directory_2 = Path(directory) / folders[1]
    return directory_1, directory_2


def get_sub_directories(directory):
    directory_list = []
    broken_directory, healthy_directory = get_folder_list(directory)
    broken_images_directory, broken_labels_directory = get_folder_list(broken_directory)
    directory_list.append(tuple([broken_images_directory, broken_labels_directory]))
    healthy_images_directory, healthy_labels_directory = get_folder_list(healthy_directory)
    directory_list.append(tuple([healthy_images_directory, healthy_labels_directory]))
    return directory_list


def split_data(number_of_data_selection, source_img_dir, source_label_dir, destination_img_dir, destination_label_dir):
    images = []
    labels = []

    # select random data
    data = random.sample([f for f in os.listdir(source_img_dir)], number_of_data_selection)
    images.extend(data)

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


def query_based_selection(total_broken_images, total_healthy_images, sub_dir_list):
    # Create query selection temporary folder
    broken_dir = QUERY_SELECT_DIR + "Broken/"
    create_sub_folders(broken_dir)

    source_broken_img_dir = sub_dir_list[0][0]
    source_broken_lbl_dir = sub_dir_list[0][1]
    destination_broken_img_dir, destination_broken_label_dir = get_folder_list(broken_dir)
    split_data(total_broken_images, source_broken_img_dir, source_broken_lbl_dir,
               destination_broken_img_dir, destination_broken_label_dir)

    healthy_temp_dir = QUERY_SELECT_DIR + "Healthy/"
    create_sub_folders(healthy_temp_dir)

    source_healthy_img_dir = sub_dir_list[1][0]
    source_healthy_lbl_dir = sub_dir_list[1][1]
    destination_healthy_img_dir, destination_healthy_label_dir = get_folder_list(healthy_temp_dir)
    split_data(total_healthy_images, source_healthy_img_dir, source_healthy_lbl_dir,
               destination_healthy_img_dir, destination_healthy_label_dir)


def create_pool_based_data(iteration, total_broken_images, total_healthy_images, src_dir_list):
    # data split between train and test / val folders
    if iteration == 0:
        if os.path.exists(pool_train_directory) and os.path.exists(pool_test_directory):
            raise Exception('Data Pooling already done, please try to create iterative dataset')
        else:
            broken_train_directory = pool_train_directory + "Broken/"
            healthy_train_directory = pool_train_directory + "Healthy/"
            broken_test_directory = pool_test_directory + "Broken/"
            healthy_test_directory = pool_test_directory + "Healthy/"

            create_sub_folders(broken_train_directory)
            create_sub_folders(healthy_train_directory)
            create_sub_folders(broken_test_directory)
            create_sub_folders(healthy_test_directory)

            train_directory = pool_train_directory
            test_directory = pool_test_directory
            train_broken_file = "Pool_train_broken_names"
            train_healthy_file = "Pool_train_healthy_names"
            test_broken_file = "Pool_test_broken_names"
            test_healthy_file = "Pool_test_healthy_names"

    else:
        broken_train_directory = iterative_train_dir + "Broken/"
        healthy_train_directory = iterative_train_dir + "Healthy/"
        broken_test_directory = iterative_val_dir + "Broken/"
        healthy_test_directory = iterative_val_dir + "Healthy/"

        create_sub_folders(broken_train_directory)
        create_sub_folders(healthy_train_directory)
        create_sub_folders(broken_test_directory)
        create_sub_folders(healthy_test_directory)

        train_directory = iterative_train_dir
        test_directory = iterative_val_dir
        train_broken_file = "Iteration_" + str(iteration) + "_train_broken_names"
        train_healthy_file = "Iteration_" + str(iteration) + "_train_healthy_names"
        test_broken_file = "Iteration_" + str(iteration) + "_val_broken_names"
        test_healthy_file = "Iteration_" + str(iteration) + "_val_healthy_names"

    # calculate training data (80%)
    number_of_broken_train_data = int(total_broken_images * train_data_ratio)
    number_of_healthy_train_data = int(total_healthy_images * train_data_ratio)

    # calculate test data (20%)
    number_of_broken_test_data = int(total_broken_images * test_data_ratio)
    number_of_healthy_test_data = int(total_healthy_images * test_data_ratio)

    source_broken_img_dir = src_dir_list[0][0]
    source_broken_label_dir = src_dir_list[0][1]

    source_healthy_img_dir = src_dir_list[1][0]
    source_healthy_label_dir = src_dir_list[1][1]

    """ Create Training Pool data for Healthy and Broken folder """
    # Train data for broken
    destination_img_dir, destination_label_dir = get_folder_list(broken_train_directory)
    filename_list = split_data(number_of_broken_train_data, source_broken_img_dir, source_broken_label_dir,
                               destination_img_dir, destination_label_dir)
    write_selected_files(train_directory, train_broken_file, filename_list)

    # Train data for healthy
    destination_img_dir, destination_label_dir = get_folder_list(healthy_train_directory)
    filename_list = split_data(number_of_healthy_train_data, source_healthy_img_dir, source_healthy_label_dir,
                               destination_img_dir, destination_label_dir)
    write_selected_files(train_directory, train_healthy_file, filename_list)

    """ Create Test Pool data for Healthy and Broken folder """
    # Test data for Broken
    destination_img_dir, destination_label_dir = get_folder_list(broken_test_directory)
    filename_list = split_data(number_of_broken_test_data, source_broken_img_dir, source_broken_label_dir,
                               destination_img_dir, destination_label_dir)
    write_selected_files(test_directory, test_broken_file, filename_list)

    # Test data for healthy
    destination_img_dir, destination_label_dir = get_folder_list(healthy_test_directory)
    filename_list = split_data(number_of_healthy_test_data, source_healthy_img_dir, source_healthy_label_dir,
                               destination_img_dir, destination_label_dir)
    write_selected_files(test_directory, test_healthy_file, filename_list)


def file_transfer(iteration):
    destination_folder_list = []
    final_dataset_root_dir = str("ACTIVE_LEARNING_DATASETS/iteration_{}_dataset").format(iteration)
    train_dir = final_dataset_root_dir + "/train/"
    destination_folder_list.append(train_dir)
    valid_dir = final_dataset_root_dir + "/valid/"
    destination_folder_list.append(valid_dir)
    test_dir = final_dataset_root_dir + "/test/"
    destination_folder_list.append(test_dir)

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
    for i in range(len(destination_folder_list)):
        for j in range(flag1, flag2):
            print("-------------")
            print("src = {} , dst = {} ".format(source_folder_list[j], destination_folder_list[i]))
            shutil.copytree(source_folder_list[j], destination_folder_list[i], symlinks=False, ignore=None,
                            copy_function=shutil.copy, ignore_dangling_symlinks=False, dirs_exist_ok=True)
            flag1 += 1
        flag2 += 2

    print("Transfer data successfully")


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

        iteration = args.iteration
        directory = args.inputDir

        if iteration == 0:
            sub_dir_list = get_sub_directories(directory)
            total_broken_images = get_images_and_labels(sub_dir_list[0][0], sub_dir_list[0][1])
            print("Broken images and labels (both) are : ", total_broken_images)

            total_healthy_images = get_images_and_labels(sub_dir_list[1][0], sub_dir_list[1][1])
            print("Healthy images and labels (both) are : ", total_healthy_images)

            create_pool_based_data(iteration, total_broken_images, total_healthy_images, sub_dir_list)
        else:
            file = ROOT_DIR + "check_iteration.txt"
            if not os.path.isfile(file):
                f = open(file, "x")
                f.close()

            if iteration == 1:
                selected_ratio = 0.15
            else:
                selected_ratio = 0.05

            # check iteration
            numbers = []
            with open(file) as f:
                for line in f.readlines():
                    if line != '\n':
                        val = line.split()
                        numbers.append(int(val[0]))
                    else:
                        f.close()

            length = len(numbers)
            if iteration in numbers:
                msg = 'Iteration {} has run before. Please select iteration {}.'.format(iteration,
                                                                                        numbers[length - 1] + 1)
                raise Exception(msg)
            else:
                f = open(file, "a+")
                f.write('%d\n' % iteration)
                f.close()

                print("********** Iteration: {} **********".format(iteration))
                print("-------------------------")

                pool_dir_list = get_sub_directories(pool_train_directory)
                total_broken_images = get_images_and_labels(pool_dir_list[0][0], pool_dir_list[0][1])
                print("Training pool broken images and labels (both) are : ", total_broken_images)

                total_healthy_images = get_images_and_labels(pool_dir_list[1][0], pool_dir_list[1][1])
                print("training pool healthy images and labels (both) are : ", total_healthy_images)

                num_of_pool_based_broken_images = int(total_broken_images * selected_ratio)
                num_of_pool_based_healthy_images = int(total_healthy_images * selected_ratio)

                # Create random query selective dataset
                query_based_selection(num_of_pool_based_broken_images, num_of_pool_based_healthy_images, pool_dir_list)

                query_based_dir = get_sub_directories(QUERY_SELECT_DIR)
                # fetch data from query selected folder to iteration dataset folder
                total_query_broken_images = get_images_and_labels(query_based_dir[0][0], query_based_dir[0][1])
                print("Query base selected broken images and labels (both) are : ", total_query_broken_images)

                total_query_healthy_images = get_images_and_labels(query_based_dir[1][0], query_based_dir[1][1])
                print("Query base selected healthy images and labels (both) are : ", total_query_healthy_images)

                create_pool_based_data(iteration, total_query_broken_images, total_query_healthy_images,
                                       query_based_dir)

                print("-------------------------")
                print("Split Successful. Current details:: ")
                # 2 is multiplied for counting the healthy data
                total_train_broken_images = get_images_and_labels(iterative_train_dir + 'Broken/images/',
                                                                  iterative_train_dir + 'Broken/labels/')
                total_train_healthy_images = get_images_and_labels(iterative_train_dir + 'Healthy/images/',
                                                                   iterative_train_dir + 'Healthy/labels/')
                print('Training: ', total_train_broken_images + total_train_healthy_images)

                total_val_broken_images = get_images_and_labels(iterative_val_dir + 'Broken/images/',
                                                                iterative_val_dir + 'Broken/labels/')
                total_val_healthy_images = get_images_and_labels(iterative_val_dir + 'Healthy/images/',
                                                                 iterative_val_dir + 'Healthy/labels/')
                print('Validation: ', total_val_broken_images + total_val_healthy_images)

                total_test_broken_images = get_images_and_labels(ROOT_DIR + 'test/Broken/images/',
                                                                 ROOT_DIR + 'test/Broken/labels')
                total_test_healthy_images = get_images_and_labels(ROOT_DIR + 'test/Healthy/images/',
                                                                 ROOT_DIR + 'test/Healthy/labels/')
                print('Testing: ',total_test_broken_images + total_test_healthy_images)
                print("-------------------------")
                
                print("File Transferring:....")
                file_transfer(iteration)

    except AssertionError as msg:
        print(msg)


if __name__ == "__main__":
    main()
