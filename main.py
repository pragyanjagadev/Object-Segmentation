import torch
# import torch.nn as nn
# import unet
import os.path
from PIL import Image
import numpy as np
from os import getcwd
# from engine import train_one_epoch, evaluate
import utils
from functions import labels2rgb, gen_lookuptable, text_segment
from rules import GenrateRules
from data_loader import IAMDataLoader
from augment import ImageAugment
from torchvision.transforms import transforms
#from masks import RuledDataset
import cv2


from Unet import build_unet

import pytorch_lightning as pl

if __name__ == "__main__":
    cwd = getcwd()

    iam_data = cwd + '/formsI-Z'
    cropped_data = cwd + '/Cropped_data'
    iam_ruled_data = cwd + '/Ruled_data'
    iam_augmented_data = cwd + '/Augmented_data'
    iam_segmented_data = cwd + '/Segmented_data'

    text_segment(iam_ruled_data)

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    labels = np.arange(256).astype(np.uint8)[np.newaxis, :]
    lut = gen_lookuptable()
    rgb = labels2rgb(labels, lut)
    #print((lut))
    #print(rgb)


    # use our dataset and defined transformations
    train_dataset = IAMDataLoader(iam_ruled_data, iam_segmented_data)
    #print(train_dataset)


    #call augment class
    obj_aug = ImageAugment(iam_data, cropped_data, iam_ruled_data, iam_augmented_data )
    #obj_aug.createAugmentedImage()

    obj_rule = GenrateRules(iam_augmented_data, cropped_data, iam_ruled_data, iam_segmented_data)
    #obj_rule.get_rules1()

    obj_mask = GenrateRules(iam_data, cropped_data, iam_ruled_data, iam_segmented_data)
    #obj_mask.get_masks()


    #obj_load_data = DataLoader(iam_ruled_data , iam_segmented_data)
    #obj_load_data()
    #obj_load_data.train_test_dataloader()

    #obj_mask = RuledDataset(iam_data, cropped_data, iam_ruled_data)
    # obj_rule.get_rules1()

    filename = iam_ruled_data + '/j01-045.png'
    img = cv2.imread(filename)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    color_dict_HSV = {'black': [[180, 255, 30], [0, 0, 0]],
                      'white': [[180, 18, 255], [0, 0, 231]],
                      'red1': [[180, 255, 255], [159, 50, 70]],
                      'red2': [[9, 255, 255], [0, 50, 70]],
                      'green': [[89, 255, 255], [36, 50, 70]],
                      'blue': [[128, 255, 255], [90, 50, 70]],
                      'yellow': [[35, 255, 255], [25, 50, 70]],
                      'purple': [[158, 255, 255], [129, 50, 70]],
                      'orange': [[24, 255, 255], [10, 50, 70]],
                      'gray': [[180, 18, 230], [0, 0, 40]]}
    lower_black = np.array([0, 42, 0])
    upper_black = np.array([179, 255, 255])


    # Create a mask. Threshold the HSV image to get only yellow colors
    #mask = cv2.inRange(hsv, lower_black, upper_black)
    #cv2.imshow('Mask', mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #test(iam_ruled_data, iam_segmented_data, filename)

    origimg1 = cropped_data + '/k02-093.png'  #m01-131.png k03-180 k04-000
    filename = iam_ruled_data + '/1k04-061_test.png'
    im = np.array(Image.open(origimg1))
    #lineHeights(im, filename, False, True)

    # inputs = torch.randn((2, 32, 256, 256))
    # e = encoder_block(32, 64)
    # x, p = e(inputs)
    # print(x.shape, p.shape)
    #
    # d = decoder_block(64, 32)
    # y = d(p, x)
    # print(y.shape)

    #inputs = torch.randn((2, 3, 512, 512))
    #model = build_unet()
    #y = model(inputs)
    #print(y.shape)

    #########################################

    #obj = IAMData(iam_ruled_data)
    #obj.setup()

    #autoencoder = AutoEncoder(representation_size=128)
    #mnist_dm = MNISTDataModule()
    #model = SimpleAutoEncoder(input_shape=mnist_dm.size(), representation_size=128)
    # We use 16-bit precision for lesser memory usage.
    # progress_bar_refresh_rate=5, to avoid Colab from crashing
    #trainer = pl.Trainer(gpus=1, max_epochs=5, precision=16, progress_bar_refresh_rate=5)
    #trainer.fit(model, mnist_dm)



