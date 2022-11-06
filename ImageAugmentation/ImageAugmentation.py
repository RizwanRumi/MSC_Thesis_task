import cv2
import os
import glob
import imutils
import time
from pathlib import Path
import numpy as np


def flip(image):
    flip_list = []
    title_list = []

    flipHorizontal = cv2.flip(image, 1)
    flip_list.append(flipHorizontal)
    title_list.append('horizontal')

    flipVertical = cv2.flip(image, 0)
    flip_list.append(flipVertical)
    title_list.append('vertical')

    return flip_list, title_list


def rotation(image):
    rotate_list = []
    title_list = []

    rotated_90_clockwise = cv2.rotate(src=image, rotateCode=cv2.ROTATE_90_CLOCKWISE)
    rotate_list.append(rotated_90_clockwise)
    title_list.append('90_clock')

    rotated_90_counterclockwise = cv2.rotate(src=image, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
    rotate_list.append(rotated_90_counterclockwise)
    title_list.append('90_counterclock')

    rotated_180 = cv2.rotate(src=image, rotateCode=cv2.ROTATE_180)
    rotate_list.append(rotated_180)
    title_list.append('180clock')

    return rotate_list, title_list

def bound_rotation(image):
    bound_list = []
    title_list = []

    for angle in range(-120, 125, 60):
        if angle != 0:
            rotated_bound = imutils.rotate_bound(image, angle=angle)
            bound_list.append(rotated_bound)
            title_list.append(str(angle) + '_bound')

    return bound_list, title_list

def grayscale(image):
    gray_list = []
    title_list = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_list.append(np.asarray(gray))
    title_list.append('gray')
    return gray_list, title_list

def colorNegative(image):
    colorneg_list = []
    title_list = []
    color_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    colored_negative = abs(255 - color_img)
    colorneg_list.append(np.asarray(colored_negative))
    title_list.append('color_neg')
    return colorneg_list, title_list

def grayNegative(image):
    grayneg_list = []
    title_list = []
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_neg = abs(255 - gray_img)
    grayneg_list.append(np.asarray(gray_neg))
    title_list.append('gray_neg')
    return grayneg_list, title_list


if __name__ == "__main__":

    start_time = time.time()
    path = "./Dataset/"
    folder_list = os.listdir(path)

    augment_list = ['Flip', 'Rotate', 'Rotate_bound', 'Gray', 'ColorNegative', 'GrayNegative']
    ext = ['jpg', 'jpeg']

    folders = [(x, y) for x in augment_list for y in folder_list if y != "Augmented"]

    for folder in folders:
        augment = folder[0]
        data_folder = folder[1]

        # create augmented folder directory
        aug_folder = path + "Augmented/" + data_folder + "_" + augment + "/"
        if not os.path.exists(aug_folder):
            os.makedirs(aug_folder)

        # get original images directory
        img_dir = "{0}{1}/".format(path, data_folder)
        files = []
        [files.extend(glob.glob(img_dir + '*.' + e)) for e in ext]

        # read all original images
        images = [cv2.imread(file) for file in files]
        total_image = len(images)

        aug_img_list = []
        labels = []

        for i in range(total_image):

            if 'Flip' == augment:
                # Flip image augmentation
                flip_images, titles = flip(images[i])
                aug_img_list = flip_images
                labels = titles

            elif 'Rotate' == augment:
                rotate_images, titles = rotation(images[i])
                aug_img_list = rotate_images
                labels = titles

            elif 'Rotate_bound' == augment:
                bound_images, titles = bound_rotation(images[i])
                aug_img_list = bound_images
                labels = titles

            elif 'Gray' == augment:
                gray_images, titles = grayscale(images[i])
                aug_img_list = gray_images
                labels = titles

            elif 'ColorNegative' == augment:
                colneg_images, titles = colorNegative(images[i])
                aug_img_list = colneg_images
                labels = titles

            elif 'GrayNegative' == augment:
                grayneg_images, titles = grayNegative(images[i])
                aug_img_list = grayneg_images
                labels = titles

            img_name = Path(files[i]).stem

            for j in range(len(labels)):
                # save augmented images in own directories
                cv2.imwrite(os.path.join(aug_folder, img_name + '_' + labels[j] + '.jpg'), aug_img_list[j])

            aug_img_list = []
            labels = []

    end_time = time.time()
    time_elapsed = np.round(end_time - start_time, 4)
    print("\n Total Processing time: ", time_elapsed, "seconds.\n")

    print("**** Finish to create augmented images ****")

