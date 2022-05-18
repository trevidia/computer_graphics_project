from os import path, listdir, mkdir, rename
from pathlib import Path
import shutil

import matplotlib
import matplotlib.pyplot as plt

import cv2 as cv

matplotlib.use('TkAgg')

ASSETS_DIR = '../assets'  # sir you would need an asset folder that contains your original images
DATASET_DIR = './dataset'  # then also this is the folder where the images would be
# arranged and also it resides in a lib folder

DATASET_SUBDIR_PREFIX = "set_"
images = listdir(ASSETS_DIR)
h_bins = 50
s_bins = 60
histSize = [h_bins, s_bins]
h_ranges = [0, 180]
s_ranges = [0, 256]
ranges = h_ranges + s_ranges
channels = [0, 1]

fig = plt.figure(figsize=(10, 7))
rows = 5
columns = 2

# loop to rename assets and arrange for comparison
count = 1
for image_index in range(0, 5):
    # Creates the dataset subdir title and
    # create the dir based on the title and path
    # if the dir exists it doesn't create the dir
    image = images[image_index]
    asset_path = path.join(ASSETS_DIR, image)
    data_dir_title = DATASET_SUBDIR_PREFIX + str(image_index + 1)
    data_dir_path = path.join(DATASET_DIR, data_dir_title)

    if not path.exists(data_dir_path):
        # after making the data dir it renames the
        # file to "original.[extension]" to make the triplicate
        # based on the original file
        print(f"Image: {image_index + 1}")
        mkdir(data_dir_path)
        shutil.copy2(asset_path, data_dir_path)
        data_set = Path(path.join(data_dir_path, image))
        new_original_path = path.join(data_dir_path, "original." + image.split(".")[1])
        data_set.rename(new_original_path)

        original = cv.imread(new_original_path)
        new_image = cv.blur(original, (100, 100))
        new_image_path = path.join(data_dir_path, "copy." + image.split(".")[1])
        cv.imwrite(new_image_path, new_image)

        hsv_original = cv.cvtColor(original, cv.COLOR_BGR2HSV)
        hsv_copy = cv.cvtColor(new_image, cv.COLOR_BGR2HSV)

        hist_original = cv.calcHist([hsv_original], channels, None, histSize, ranges, accumulate=False)
        cv.normalize(hist_original, hist_original, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

        hist_copy = cv.calcHist([hsv_copy], channels, None, histSize, ranges, accumulate=False)
        cv.normalize(hist_copy, hist_copy, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
        compare_method = cv.HISTCMP_CORREL

        # this line compares the lines
        original_copy = cv.compareHist(hist_original, hist_copy, compare_method)

        print(f"Similarity based on original: {new_original_path}"
              f" and copy: {new_image_path} is: {original_copy}\n"
              f"Is not similar: {0 <= original_copy <= 0.5} \n"
              f"Is similar: {0.5 < original_copy <= 1} \n")
        fig.add_subplot(rows, columns, count)
        plt.title("Original")
        plt.imshow(cv.cvtColor(original, cv.COLOR_BGR2RGB))
        plt.axis("off")

        count += 1
        fig.add_subplot(rows, columns, count)
        plt.title("Copy")
        plt.imshow(cv.cvtColor(new_image, cv.COLOR_BGR2RGB))
        plt.axis("off")
        count += 1

plt.show()
