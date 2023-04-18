import cv2
import numpy as np
import random
import os
from PIL import Image
from functions import lineHeights


class GenrateRules:
    def __init__(self, iam_data_path, cropped_data_path, iam_ruled_data_path, iam_segmented_data):

        self.iam_data_path = iam_data_path
        self.cropped_data_path = cropped_data_path
        self.iam_ruled_data_path = iam_ruled_data_path
        self.iam_segmented_data = iam_segmented_data

    def projection_analysis(self, im):
        # compute the ink density histogram (sum each rows)
        hist = cv2.reduce(im, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32F)
        hist = hist.ravel()
        # find peaks withing the ink density histogram
        max_hist = max(hist)
        mean_hist = np.mean(hist)
        thres_hist = mean_hist / max_hist
        peaks = peakutils.indexes(hist, thres=thres_hist, min_dist=50)
        # find peaks that are too high
        mean_peaks = np.mean(hist[peaks])
        std_peaks = np.std(hist[peaks])
        thres_peaks_high = mean_peaks + 1.5 * std_peaks
        thres_peaks_low = mean_peaks - 3 * std_peaks
        peaks = peaks[np.logical_and(hist[peaks] < thres_peaks_high, hist[peaks] > thres_peaks_low)]

        return peaks

    def center_crop(self, img, file):

        cropped_file = self.cropped_data_path + '/' + file
        width = img.shape[1]
        height = img.shape[0]

        new_width = min(width, height)
        new_height = min(width, height)

        left = int(np.ceil((width - new_width) / 2))
        right = width - int(np.floor((width - new_width) / 2))

        top = int(np.ceil((height - new_height) / 1.5))
        bottom = height - int(np.floor((height - new_height)))

        if len(img.shape) == 2:
            center_cropped_img = img[top:bottom, left:right]
        else:
            center_cropped_img = img[top:bottom, left:right, ...]

        cv2.imwrite(cropped_file, center_cropped_img)

    def get_masks(self, show=False, write=False, return_img=False):
        i = 0
        no_line = 0
        for subdir, dirs, files in os.walk(self.iam_ruled_data_path):
            for file in files:
                if file != '.DS_Store':
                    # if (i < 1):
                    # i = i + 1
                    # print(os.path.join(subdir, file))
                    input_file = self.iam_ruled_data_path + '/' + file
                    output_file = self.iam_segmented_data + '/' + file
                    img = cv2.imread(input_file)
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    print(input_file)
                    lower_black = np.array([0, 42, 0])
                    upper_black = np.array([179, 255, 255])
                    mask = cv2.inRange(hsv, lower_black, upper_black)
                    cv2.imwrite(output_file,mask)

    def get_rules1(self, show=False, write=False, return_img=False):
        i = 0
        no_line = 0
        for subdir, dirs, files in os.walk(self.iam_data_path):
            for file in files:
                if file != '.DS_Store':
                    #if (i < 1):
                    #i = i + 1
                    # print(os.path.join(subdir, file))
                    input_file = self.iam_data_path + '/' + file
                    output_file = self.iam_ruled_data_path + '/' + file
                    img = cv2.imread(input_file)
                    print(input_file)
                    print(img.shape[1])
                    print(img.shape[0])
                    # croping image
                    self.center_crop(img, file)
                    cropped_file = self.cropped_data_path + '/' + file
                    cropped_img = cv2.imread(cropped_file)
                    print(cropped_file)
                    print(output_file)
                    origimg = np.array(Image.open(cropped_file))

                    lineHeights(origimg, output_file, False, True)

    def get_rules(self):

        i = 0
        no_line = 0
        for subdir, dirs, files in os.walk(self.iam_data_path):
            for file in files:
                #if (i < 1):
                    #i = i + 1
                print(file)
                # print(os.path.join(subdir, file))
                input_file = self.iam_data_path + '/' + file
                output_file = self.iam_ruled_data_path + '/' + file
                img = cv2.imread(input_file)

                # croping image
                self.center_crop(img, file)
                cropped_file = self.cropped_data_path + '/' + file
                cropped_img = cv2.imread(cropped_file)

                img_height, img_width = cropped_img.shape[:-1]
                grayImage = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

                GaussianFilter = cv2.GaussianBlur(grayImage, (5, 5), 0)
                _, binarizedImage = cv2.threshold(GaussianFilter, 127, 255, cv2.THRESH_BINARY)
                unique, counts = np.unique(binarizedImage, return_counts=True)

                binarizedImage[binarizedImage == 0] = 1
                binarizedImage[binarizedImage == 255] = 0
                unique, counts = np.unique(binarizedImage, return_counts=True)

                horizontal_projection = np.sum(binarizedImage, axis=1)

                height, width = binarizedImage.shape  #

                blankImage = np.zeros((height, width, 3), np.uint8)
                prev = 0
                prev_1 = 0

                line_width_bool = random.choice([False, True])
                line_width = 3 if line_width_bool else 2

                for row in range(height):

                    prev_1 = prev
                    prev = (horizontal_projection[row] * width / height)

                    #if (np.floor((prev_1)) > 1) & (prev <= 0):
                    if (int(prev_1) != 0) & (prev <= 0):
                        cv2.line(cropped_img, (0, row), (width, row), (0, 0, 0), 2)
                        no_line = no_line + 1

                    cv2.imwrite(output_file, cropped_img)
