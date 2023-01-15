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

Pool_root_dir = "./Training_pool_dataset/"
pool_train_directory = Pool_root_dir + "train/"
pool_test_directory = Pool_root_dir + "test/"

Query_selection_dir = "./Query_selection_dataset/"

Iteration_root_dir = "./Iteration_dataset/"
iterative_train_dir = Iteration_root_dir + "train/"
iterative_val_dir = Iteration_root_dir + "valid/"

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

def check_precision(number):
    # print("Precision_value", number)
    fstr = repr(number)
    sign_digit, fractional_digit = fstr.split('.')
    if int(fractional_digit) >= 5:
        round_number = int(sign_digit) + 1
    else:
        round_number = int(sign_digit)
    return round_number

def print_message(message, t_number):
    print("{} {}".format(message, t_number))

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
    else:
        raise Exception("Text files are missing..")
    # return images, labels
    return names

def query_based_selection(total_broken_images, total_healthy_images, sub_dir_list):
    # Create query selection temporary folder
    broken_dir = Query_selection_dir + "Broken/"
    create_sub_folders(broken_dir)

    source_broken_img_dir = sub_dir_list[0][0]
    source_broken_lbl_dir = sub_dir_list[0][1]
    destination_broken_img_dir, destination_broken_label_dir = get_folder_list(broken_dir)
    split_data(total_broken_images, source_broken_img_dir, source_broken_lbl_dir,
               destination_broken_img_dir, destination_broken_label_dir)

    healthy_temp_dir = Query_selection_dir + "Healthy/"
    create_sub_folders(healthy_temp_dir)

    source_healthy_img_dir = sub_dir_list[1][0]
    source_healthy_lbl_dir = sub_dir_list[1][1]
    destination_healthy_img_dir, destination_healthy_label_dir = get_folder_list(healthy_temp_dir)
    split_data(total_healthy_images, source_healthy_img_dir, source_healthy_lbl_dir,
               destination_healthy_img_dir, destination_healthy_label_dir)

def train_test_split(iteration, total_broken_images, total_healthy_images, src_dir_list):
    # data split between train and test / val folders
    if iteration == 0:
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
    broken_train_data = round(total_broken_images * train_data_ratio, 2)
    number_of_broken_train_data = check_precision(broken_train_data)
    # print("number_of_broken_train_data: ", number_of_broken_train_data)
    healthy_train_data = round(total_healthy_images * train_data_ratio, 2)
    number_of_healthy_train_data = check_precision(healthy_train_data)
    # print("number_of_healthy_train_data: ", number_of_healthy_train_data)

    # calculate test data (20%)
    # broken_test_data = round(total_broken_images * test_data_ratio, 2)
    # number_of_broken_test_data = check_precision(broken_test_data)
    number_of_broken_test_data = total_broken_images - number_of_broken_train_data
    # print("number_of_broken_test_data: ", number_of_broken_test_data)
    # healthy_test_data = round(total_healthy_images * test_data_ratio, 2)
    # number_of_healthy_test_data = check_precision(healthy_test_data)
    number_of_healthy_test_data = total_healthy_images - number_of_healthy_train_data
    # print("number_of_healthy_test_data: ", number_of_healthy_test_data)

    source_broken_img_dir = src_dir_list[0][0]
    source_broken_label_dir = src_dir_list[0][1]

    source_healthy_img_dir = src_dir_list[1][0]
    source_healthy_label_dir = src_dir_list[1][1]

    print("-------------------------")
    # Train data for broken
    destination_img_dir, destination_label_dir = get_folder_list(broken_train_directory)
    train_broken_file_list = split_data(number_of_broken_train_data, source_broken_img_dir, source_broken_label_dir,
                                        destination_img_dir, destination_label_dir)
    write_selected_files(train_directory, train_broken_file, train_broken_file_list)
    print_message("Train broken data: ", len(train_broken_file_list))

    # Test / valid data for broken
    destination_img_dir, destination_label_dir = get_folder_list(broken_test_directory)
    test_broken_file_list = split_data(number_of_broken_test_data, source_broken_img_dir, source_broken_label_dir,
                                       destination_img_dir, destination_label_dir)
    write_selected_files(test_directory, test_broken_file, test_broken_file_list)
    print_message("Test broken data: ", len(test_broken_file_list))

    # Train data for healthy
    destination_img_dir, destination_label_dir = get_folder_list(healthy_train_directory)
    train_healthy_file_list = split_data(number_of_healthy_train_data, source_healthy_img_dir, source_healthy_label_dir,
                                         destination_img_dir, destination_label_dir)
    write_selected_files(train_directory, train_healthy_file, train_healthy_file_list)
    print_message("Train healthy data: ", len(train_healthy_file_list))

    # Test / valid data for healthy
    destination_img_dir, destination_label_dir = get_folder_list(healthy_test_directory)
    test_healthy_file_list = split_data(number_of_healthy_test_data, source_healthy_img_dir, source_healthy_label_dir,
                                        destination_img_dir, destination_label_dir)
    write_selected_files(test_directory, test_healthy_file, test_healthy_file_list)
    print_message("Test healthy data: ", len(test_healthy_file_list))
    print("-------------------------")

def file_transfer(iteration):
    destination_folder_list = []
    final_dataset_root_dir = str("Active_Learning_Datasets/iteration_{}_dataset").format(iteration)
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
            shutil.copytree(source_folder_list[j], destination_folder_list[i], symlinks=False, ignore=None,
                            copy_function=shutil.copy, ignore_dangling_symlinks=False, dirs_exist_ok=True)
            flag1 += 1
        flag2 += 2

    print("Transfer data successfully in Active_Learning_Datasets")


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
        default=3
    )

    try:
        args = parser.parse_args()
        assert os.path.isdir(args.inputDir), "The directory is not found. Please check the directory. " \
                                             "Hints: run at first -dir ./test_dataset -I 0"

        iteration = args.iteration
        directory = args.inputDir

        if iteration == 0:
            sub_dir_list = get_sub_directories(directory)
            total_broken_images = get_images_and_labels(sub_dir_list[0][0], sub_dir_list[0][1])
            print("Broken images and labels (both) are : ", total_broken_images)

            total_healthy_images = get_images_and_labels(sub_dir_list[1][0], sub_dir_list[1][1])
            print("Healthy images and labels (both) are : ", total_healthy_images)

            """ Create Training Pool data for Healthy and Broken folder """

            print("\n Training Pool dataset: ")
            if total_broken_images > 2 and total_healthy_images > 2:
                train_test_split(iteration, total_broken_images, total_healthy_images, sub_dir_list)
            else:
                raise Exception("There have no sufficient image and label files. Please try to create iterative dataset")
        else:
            if directory == "./test_dataset" and "./test_dataset/":
                msg = "Please change the input directory and replace by ./Training_pool_dataset"
                raise Exception(msg)
            else:

                file = Pool_root_dir + "check_iteration.txt"
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
                    print("********** Iteration: {} **********".format(iteration))

                    pool_dir_list = get_sub_directories(pool_train_directory)
                    total_broken_images = get_images_and_labels(pool_dir_list[0][0], pool_dir_list[0][1])
                    print("Training pool broken images and labels (both) are : ", total_broken_images)

                    total_healthy_images = get_images_and_labels(pool_dir_list[1][0], pool_dir_list[1][1])
                    print("training pool healthy images and labels (both) are : ", total_healthy_images)

                    # select number of percentage data
                    pool_based_broken_images = round(total_broken_images * selected_ratio, 2)
                    num_of_pool_based_broken_images = check_precision(pool_based_broken_images)
                    pool_based_healthy_images = round(total_healthy_images * selected_ratio, 2)
                    num_of_pool_based_healthy_images = check_precision(pool_based_healthy_images)

                    if num_of_pool_based_broken_images > 2 and num_of_pool_based_healthy_images > 2:
                        # track iteration
                        f = open(file, "a+")
                        f.write('%d\n' % iteration)
                        f.close()

                        print(str(selected_ratio * 100) + "% random data has taken to split the train & val(80,20) data")

                        # Create random query selective dataset
                        query_based_selection(num_of_pool_based_broken_images, num_of_pool_based_healthy_images, pool_dir_list)

                        query_based_dir = get_sub_directories(Query_selection_dir)

                        print("-------------------------")
                        print("Query selection data: ")

                        # fetch data from query selected folder to iteration dataset folder
                        total_query_broken_images = get_images_and_labels(query_based_dir[0][0], query_based_dir[0][1])
                        print("Query base selected broken images and labels (both) are : ", total_query_broken_images)

                        total_query_healthy_images = get_images_and_labels(query_based_dir[1][0], query_based_dir[1][1])
                        print("Query base selected healthy images and labels (both) are : ", total_query_healthy_images)

                        train_test_split(iteration, total_query_broken_images, total_query_healthy_images, query_based_dir)

                        print("Data has been split successfully. Current details: ")

                        # calculate total split data
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

                        total_test_broken_images = get_images_and_labels(Pool_root_dir + 'test/Broken/images/',
                                                                         Pool_root_dir + 'test/Broken/labels')
                        total_test_healthy_images = get_images_and_labels(Pool_root_dir + 'test/Healthy/images/',
                                                                          Pool_root_dir + 'test/Healthy/labels/')
                        print('Testing: ', total_test_broken_images + total_test_healthy_images)
                        print("-------------------------")

                        print("File Transferring:....")
                        file_transfer(iteration)
                        # delete query selection directory
                        shutil.rmtree(Query_selection_dir)
                    else:
                        print("There have no sufficient data. Stop iteration and add new data in test_dataset.")

    except AssertionError as msg:
        print(msg)


if __name__ == "__main__":
    main()
