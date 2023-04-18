import cv2
import numpy as np
from PIL import Image
import torch
# import transforms as T
#
#
# def get_transform(train):
#     transforms = []
#     transforms.append(T.PILToTensor())
#     transforms.append(T.ConvertImageDtype(torch.float))
#     if train:
#         transforms.append(T.RandomHorizontalFlip(0.5))
#     return T.Compose(transforms)

def text_segment(img_dir):

    image = img_dir + '/j01-042.png'
    print(image)

    img_bgr = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    # Convert the image into HSV
    hsv_image = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2HSV)

    # Set the blue range
    lower_black = np.array([0, 42, 0])
    upper_black = np.array([179, 255, 255])

    # Apply the blue mask
    mask = cv2.inRange(hsv_image, lower_black, upper_black)

    # Set a white range
    light_white = (0, 0, 200)
    dark_white = (145, 60, 255)

    # Apply the white mask
    mask_white = cv2.inRange(hsv_image, light_white, dark_white)
    # Combine the two masks
    final_mask = mask + mask_white
    result = cv2.bitwise_and(img_bgr, img_bgr, mask=final_mask)
    i3 = Image.open(result)
    i3.show()
    # Clean up the segmentation using a blur
    blur = cv2.GaussianBlur(result, (7, 7), 0)

    return blur


def gen_lookuptable():
    """
    Generate a label colormap compatible with opencv lookup table, based on
    Rick Szelski algorithm in `Computer Vision: Algorithms and Applications`,
    appendix C2 `Pseudocolor Generation`.
    :Returns:
      color_lut : opencv compatible color lookup table
    """
    tobits = lambda x, o: np.array(list(np.binary_repr(x, 24)[o::-3]), np.uint8)
    arr = np.arange(256)
    r = np.concatenate([np.packbits(tobits(x, -3)) for x in arr])
    g = np.concatenate([np.packbits(tobits(x, -2)) for x in arr])
    b = np.concatenate([np.packbits(tobits(x, -1)) for x in arr])
    return np.concatenate([[[b]], [[g]], [[r]]]).T


def labels2rgb(labels, lut):
    """
    Convert a label image to an rgb image using a lookup table
    :Parameters:
      labels : an image of type np.uint8 2D array
      lut : a lookup table of shape (256, 3) and type np.uint8
    :Returns:
      colorized_labels : a colorized label image
    """
    return cv2.LUT(cv2.merge((labels, labels, labels)), lut)


def lineHeights(origimg, filename, show=False, write=False, return_img=False):
    if origimg.ndim == 3:
        origimg = cv2.cvtColor(origimg, cv2.COLOR_BGR2GRAY)

    img = origimg.astype(float)
    if np.max(img) > 1:
        img /= 255.0

    mean_height = 0.0
    std_height = 1.0
    num_lines = 0
    col_range = None

    # col_range = (0,int(img.shape[1] / 4))
    # col_range = (int(img.shape[1]/8.0),int(img.shape[1] / 3.0))
    # profile = np.sum(img[:,col_range[0]:col_range[1]], axis=1)
    col_div = 3
    for i in range(col_div):
        c_col_range = (int((float(i) / col_div) * img.shape[1] * 0.45),
                       int((float(i + 1) / col_div) * img.shape[1] * 0.45))
        profile = np.sum(img[:, c_col_range[0]:c_col_range[1]], axis=1)
        # find best sigma value
        for s in range(21, 302, 10):
            c_g_profile = cv2.GaussianBlur(profile, (s, s), 0)
            # central difference
            minima = np.where((c_g_profile[1:-1] < c_g_profile[:-2]) \
                              & (c_g_profile[1:-1] < c_g_profile[2:]))[0]
            # get local minima (except first and last line):
            heights = minima[1:] - minima[:-1]
            heights = heights[1:-1]
            if len(heights) == 0:
                break
            c_mean_height = np.mean(heights)
            c_std_height = np.std(heights)

            if len(minima) > 8 and len(minima) < 50 \
                    and c_mean_height / c_std_height > mean_height / std_height:
                num_lines = len(minima) + 1
                #           and c_std_height < std_height:
                mean_height = c_mean_height
                std_height = c_std_height
                col_range = c_col_range
                g_profile = c_g_profile

        # if no proper lines found so far it's probably only
        # an excerpt of a charter
        if not col_range:
            for s in range(7, 308, 10):
                c_g_profile = cv2.GaussianBlur(profile, (s, s), 0)
                # central difference
                minima = np.where((c_g_profile[1:-1] < c_g_profile[:-2]) \
                                  & (c_g_profile[1:-1] < c_g_profile[2:]))[0]
                # get local minima (except first and last line):
                heights = minima[1:] - minima[:-1]

                if len(heights) == 0:
                    break
                c_mean_height = np.median(heights)
                c_std_height = np.std(heights)

                if c_mean_height / c_std_height > mean_height / std_height:
                    num_lines = len(minima) + 1
                    #           and c_std_height < std_height:
                    mean_height = c_mean_height
                    std_height = c_std_height
                    col_range = c_col_range
                    g_profile = c_g_profile

    if (show or write or return_img) and col_range != None:
        line_img = origimg.copy()
        # line_img = np.bitwise_not(line_img)
        line_img = cv2.cvtColor(line_img, cv2.COLOR_GRAY2BGR)
        # line_img[:, col_range[0]:col_range[1]] /= 2
        # draw line boundings
        maxima = np.where((g_profile[1:-1] > g_profile[:-2]) \
                          & (g_profile[1:-1] > g_profile[2:]))[0]
        minima = np.where((g_profile[1:-1] < g_profile[:-2]) \
                          & (g_profile[1:-1] < g_profile[2:]))[0]

        print(maxima)
        for maxi in maxima:
            color = list(np.random.random(size=3) * 256)
            cv2.line(line_img, (0, maxi), (img.shape[1], maxi), color, 3)

        # draw line masses
        minima = np.where((g_profile[1:-1] < g_profile[:-2]) \
                          & (g_profile[1:-1] < g_profile[2:]))[0]
        # for mini in minima:
        #    cv2.line(line_img, (0, mini), (img.shape[1], mini), (100, 0, 255), 3)

        show_img = cv2.resize(line_img,
                              (int((800.0 / line_img.shape[0]) * \
                                   line_img.shape[1]), 800))
        #                              show_img, 0.2, 0.2)
        if show:
            cv2.imshow('lines', show_img)
            cv2.waitKey()
        if write:
            cv2.imwrite(filename, show_img)

    if return_img:
        return num_lines, mean_height, std_height, line_img

    return num_lines, mean_height, std_height, None
