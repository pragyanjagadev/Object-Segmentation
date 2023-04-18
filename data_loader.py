import os
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from torch.nn import functional as F
from torchvision import transforms
import pytorch_lightning as pl
import splitfolders


# , batch_size=64, num_workers=4, transform =None

class IAMDataLoader(Dataset):
    def __init__(self, image_dir, mask_dir):
        print("here===")
        # self.batch_size = batch_size
        # self.num_workers = num_workers
        self.image_dataset = []
        self.train_imgs = []
        self.test_imgs = []
        self.mask_rain_imgs = []
        self.mask_test_imgs = []

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        # self.transform = transform
        # self.images = os.listdir(image_dir)
        self.images = list(sorted(os.listdir(self.image_dir)))
        self.masks = list(sorted(os.listdir(self.mask_dir)))

        # We hardcode dataset specific stuff here.
        self.size = 256
        self.num_classes = 256
        self.dims = (1, 28, 28)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #label_range = np.linspace(0, 1, 256)
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        print("classes============")
        print(num_objs)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])

        target = {}
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id

        print(target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


    def load(self, stage=None):

        images = os.listdir(self.image_dir)
        mask_images = os.listdir(self.mask_dir)
        for i, image_name in enumerate(images):
            if image_name.split('.')[1] == 'png':
                # print(self.data_dir + '/' +image_name)
                image = cv2.imread(self.image_dir + '/' + image_name, 1)
                # image = Image.fromarray(image)
                # image = image.resize((self.size, self.size))
                self.train_imgs.append(image)

        for i, image_name in enumerate(mask_images):
            if image_name.split('.')[1] == 'png':
                # print(self.data_dir + '/' +image_name)
                image = cv2.imread(self.mask_images + '/' + image_name, 1)
                # image = Image.fromarray(image)
                # image = image.resize((self.size, self.size))
                self.mask_imgs.append(image)

        print(len(self.train_imgs))
        print(len(self.mask_imgs))
        # print(len(self.image_dataset))

    def train_test_dataloader(self):
        self.train_data, self.test_data = train_test_split(self.image_dataset, test_size=0.20, random_state=0)
        # print(len(self.train_data))
        # print(len(self.test_data))
