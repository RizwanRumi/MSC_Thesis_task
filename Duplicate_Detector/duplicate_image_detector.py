import skimage.color
import numpy as np
import cv2
import os
import collections
import time
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


# Function that processes the directories that were input as parameters
def check_directory(directory):
    directory = Path(directory)
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory " + str(directory) + " does not exist")
    return directory


# Function that displays a progress bar during the search
def show_progress(count, list, task='processing images'):
    if count + 1 == len(list):
        print(f"..... {task}: [{count}/{len(list)}] [{count / len(list):.0%}]", end="\r")
        print(f"..... {task}: [{count + 1}/{len(list)}] [{(count + 1) / len(list):.0%}]")
    else:
        print(f"..... {task}: [{count}/{len(list)}] [{count / len(list):.0%}]", end="\r")


# Function that check the respective MSE value
def map_similarity(similarity):
    try:
        similarity = float(similarity)
        ref = similarity
    except:
        if similarity == "low":
            ref = 470
        # search for exact duplicate images, extremely sensitive, MSE < 0.1
        elif similarity == "high":
            ref = 0.1
        # normal, search for duplicates, recommended, MSE < 200
        else:
            ref = 200
    return ref


# Function that creates a list of matrices for each image found in the folder
def create_imgs_matrix(directory, px_size):
    # create list of files found in directory
    folder_files = [(directory, filename) for filename in os.listdir(directory)]
    # file_list = [file for filename in os.listdir(directory)]

    # create images matrix
    imgs_matrix, delete_index = [], []
    for count, file in enumerate(folder_files):

        show_progress(count, folder_files, task='preparing files')

        path = Path(file[0]) / file[1]
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if type(img) == np.ndarray:
            img = img[..., 0:3]
            img = cv2.resize(img, dsize=(px_size, px_size), interpolation=cv2.INTER_CUBIC)

            if len(img.shape) == 2:
                img = skimage.color.gray2rgb(img)
            imgs_matrix.append(img)
        else:
            delete_index.append(count)

    for index in reversed(delete_index):
        del folder_files[index]

    return imgs_matrix, folder_files

"""
# Function for rotating an image matrix by a 90 degree angle
def rotate_img(image):
    image = np.rot90(image, k=1, axes=(0, 1))
    return image
"""

# Function that calculates the mean squared error (mse) between two image matrices
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


# Function that plots two compared image files and their mse
def show_img_figs(imageA, imageB, err):
    fig = plt.figure()
    plt.suptitle("MSE: %.2f" % (err))
    # plot first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")
    # plot second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")
    # show the images
    plt.show()


# Function for printing filename info of plotted image files
def show_file_info(imageA, imageB):
    imageA = "..." + str(imageA)[-45:]
    imageB = "..." + str(imageB)[-45:]
    print(f"""Duplicate files:\n{imageA} and \n{imageB}\n""")


# Function that searches one directory for duplicate/similar images
def search_one_dir(img_matrices_A, folderfiles_A, similarity, show_output=False):
    result = {}
    lower_quality = []
    ref = similarity

    # find duplicates/similar images within one folder
    for count_A, imageMatrix_A in enumerate(img_matrices_A):
        img_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        while img_id in result.keys():
            img_id = str(int(img_id) + 1)
        show_progress(count_A, img_matrices_A, task='comparing images')
        for count_B, imageMatrix_B in enumerate(img_matrices_A):
            if count_B > count_A != len(img_matrices_A):
                err = mse(imageMatrix_A, imageMatrix_B)
                if err < ref:
                    if show_output:
                        show_img_figs(imageMatrix_A, imageMatrix_B, err)
                        # 0 is the path, 1 is the filename
                        show_file_info(Path(folderfiles_A[count_A][0]) / folderfiles_A[count_A][1],
                                       Path(folderfiles_A[count_B][0]) / folderfiles_A[count_B][1])
                    if img_id in result.keys():
                        result[img_id]["duplicates"] = result[img_id]["duplicates"] + [
                            #  with path: str(Path(folderfiles_A[count_B][0]) / folderfiles_A[count_B][1])
                            str(folderfiles_A[count_B][1])]
                    else:
                        result[img_id] = {'filename': str(folderfiles_A[count_A][1]),
                                          'location': str(Path(folderfiles_A[count_A][0]) / folderfiles_A[count_A][1]),
                                          'duplicates': [
                                              #  with path: str(Path(folderfiles_A[count_B][0]) / folderfiles_A[count_B][1])
                                              str(folderfiles_A[count_B][1])]}
    result = collections.OrderedDict(result.items())
    return result


# Function for deleting the lower quality images that were found after the search
def delete_images(image_directory, image_list):
    deleted = 0
    # delete lower quality images
    for file in image_list:
        print("\nDeletion in progress...", end="\r")
        try:
            image_path = Path(image_directory / file)
            os.remove(image_path)
            print("Deleted file:", file, end="\r")
            deleted += 1
        except:
            print("Could not delete file:", file, end="\r")
    print("\n***\nDeleted", deleted, "images.")


# Function for deleting the lower quality images with labels that were found after the search
def delete_images_with_lables(image_directory, label_directory, image_list):
    deleted = 0
    # delete lower quality images
    for file in image_list:
        print("\nDeletion in progress...", end="\r")
        # get filename
        name, ext = os.path.splitext(file)
        label = str(name + '.txt')
        image_path = Path(image_directory / file)
        label_path = Path(label_directory / label)
        try:
            os.remove(image_path)
            os.remove(label_path)
            print("Deleted file: {0}, {1}".format(file, label), end="\r")
            deleted += 1
        except:
            print("Can not Delete : {0}, {1}".format(file, label), end="\r")
    print("\n***\nDeleted", deleted, "images.")


if __name__ == "__main__":

    start_time = time.time()
    folder_path = "./Datasets/augmented/raw_flip_rotation_2x/train/"
    #folder_path = "./Datasets/Broken/"
    directory = check_directory(folder_path)

    # For only images without labels
    #images_directory = folder_path

    # for images and labels folder
    folders = os.listdir(directory)
    images_directory = Path(directory / folders[0])
    labels_directory = Path(directory / folders[1])

    img_matrices, files = create_imgs_matrix(images_directory, px_size=100)
    ref = map_similarity(similarity="low")
    duplicate_result = search_one_dir(img_matrices, files, ref, show_output=False)
    end_time = time.time()
    time_elapsed = np.round(end_time - start_time, 4)

    duplicate_images_list = set()
    # x = duplicate_result.keys()
    # print(x)
    for res in duplicate_result.values():
        # print("res: ", res)
        for x, y in res.items():
            print(x, ':', y)
        for lst in res.get("duplicates"):
            duplicate_images_list.add(lst)

    total_duplicate = len(duplicate_images_list)

    images = 'image' if total_duplicate == 1 else 'images'

    print("\nFound", total_duplicate, images, "with one or more duplicate/similar images in", time_elapsed,
          "seconds.\n")

    if total_duplicate != 0:
        # optional delete images
        usr = input("Are you sure you want to delete all duplicate images with their labels?"
                    "\nThis cannot be undone. (y/n) ")
        if str(usr) == "y":
            # delete_images(images_directory, list(duplicate_images_list))
            delete_images_with_lables(images_directory, labels_directory, list(duplicate_images_list))
        else:
            print("Image deletion canceled.")
