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
    # check if directories are valid
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

# Function that maps the similarity grade to the respective MSE value
def map_similarity(similarity):
    try:
        similarity = float(similarity)
        ref = similarity
    except:
        if similarity == "low":
            ref = 466
        # search for exact duplicate images, extremly sensitive, MSE < 0.1
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
    #file_list = [filename for filename in os.listdir(directory)]

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

# Function for rotating an image matrix by a 90 degree angle
def rotate_img(image):
    image = np.rot90(image, k=1, axes=(0, 1))
    return image

# Function that calulates the mean squared error (mse) between two image matrices
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

# Function for checking the quality of compared images, appends the lower quality image to the list
def check_img_quality(imageA, imageB):
    size_imgA = os.stat(imageA).st_size
    size_imgB = os.stat(imageB).st_size
    if size_imgA >= size_imgB:
        return imageA, imageB
    else:
        return imageB, imageA

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
            if count_B > count_A and count_A != len(img_matrices_A):
                rotations = 0
                while rotations <= 3:
                    if rotations != 0:
                        imageMatrix_B = rotate_img(imageMatrix_B)

                    err = mse(imageMatrix_A, imageMatrix_B)
                    if err < ref:
                        if show_output:
                            show_img_figs(imageMatrix_A, imageMatrix_B, err)
                            show_file_info(Path(folderfiles_A[count_A][0]) / folderfiles_A[count_A][1],
                                                # 0 is the path, 1 is the filename
                                                Path(folderfiles_A[count_B][0]) / folderfiles_A[count_B][1])
                        if img_id in result.keys():
                            result[img_id]["duplicates"] = result[img_id]["duplicates"] + [
                                str(Path(folderfiles_A[count_B][0]) / folderfiles_A[count_B][1])]
                        else:
                            result[img_id] = {'filename': str(folderfiles_A[count_A][1]),
                                              'location': str(
                                                  Path(folderfiles_A[count_A][0]) / folderfiles_A[count_A][1]),
                                              'duplicates': [
                                                  str(Path(folderfiles_A[count_B][0]) / folderfiles_A[count_B][1])]}
                        try:
                            high, low = check_img_quality(
                                Path(folderfiles_A[count_A][0]) / folderfiles_A[count_A][1],
                                Path(folderfiles_A[count_B][0]) / folderfiles_A[count_B][1])
                            lower_quality.append(str(low))
                        except:
                            pass
                        break
                    else:
                        rotations += 1

    result = collections.OrderedDict(sorted(result.items()))
    lower_quality = list(set(lower_quality))

    return result, lower_quality


if __name__ == "__main__":

    start_time = time.time()
    folder_path = "E:/Thesis_Task/Duplicate_Detector/dataset/"
    directory = check_directory(folder_path)
    img_matrices, files = create_imgs_matrix(directory, px_size=50)
    ref = map_similarity(similarity="low")
    result, lower_quality = search_one_dir(img_matrices, files, ref, show_output=False)

    if len(result) == 1:
        images = "image"
    else:
        images = "images"

    end_time = time.time()
    time_elapsed = np.round(end_time - start_time, 4)
    print("\n")

    for id in enumerate(result):
        for y in result[id[1]]:
            print(y, ':', result[id[1]][y])

    print("\nFound", len(result), images, "with one or more duplicate/similar images in", time_elapsed, "seconds.")

