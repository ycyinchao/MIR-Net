#!/usr/bin/python3
# coding=utf-8

import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

from data import transform


class CamObjDataset(Dataset):
    def __init__(self, cfg):
        self.mode = cfg.mode
        self.trainsize = cfg.trainsize
        self.data_name = cfg.val_root.split('/')[-1]

        if self.mode == 'train':
            image_root = cfg.train_root + '/Image/'
            gt_root = cfg.train_root + '/Scribble/'
        else:
            image_root = cfg.val_root + '/Image/'
            gt_root = cfg.val_root + '/GT/'
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        # sorted files
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        # filter mathcing degrees of files，不要过滤，最小的数据集里面有不匹配的两个图片
        # self.filter_files()

        # get size of dataset
        self.size = len(self.images)
        print('>>> trainig/validing with {} samples'.format(self.size))

        self.mean = np.array([[[0.485 * 256, 0.456 * 256, 0.406 * 256]]])
        self.std = np.array([[[0.229 * 256, 0.224 * 256, 0.225 * 256]]])
        if self.mode == 'train':
            self.transform = transform.Compose(transform.Normalize(mean=self.mean, std=self.std),
                                               transform.Resize(self.trainsize, self.trainsize),
                                               transform.RandomHorizontalFlip(),
                                               transform.ToTensor())

        elif self.mode == 'test':
            self.transform = transform.Compose(transform.Normalize(mean=self.mean, std=self.std),
                                               transform.Resize(self.trainsize, self.trainsize),
                                               transform.ToTensor()
                                               )
        else:
            raise ValueError

    def __getitem__(self, idx):
        imagepath, maskpath = self.images[idx], self.gts[idx]
        image = cv2.imread(imagepath).astype(np.float32)[:, :, ::-1]
        mask = cv2.imread(maskpath).astype(np.float32)[:, :, ::-1]

        H, W, C             = mask.shape
        if self.mode == 'train':
            # 0代表未标注，1代表目标部分标注，2代表背景部分标注
            image, mask = self.transform(image, mask)
            mask[mask == 0.] = 255.  # 0：代表未标注
            mask[mask == 2.] = 0.  # 2：代表背景部分标注，1代表目标部分标注
        else:
            image,_ = self.transform(image, mask)
            mask = torch.from_numpy(mask.copy()).permute(2, 0, 1)
            mask = mask.mean(dim=0, keepdim=True)
            mask /= 255

        return image, mask, (H, W), maskpath.split('/')[-1]

    def __len__(self):
        return self.size

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def mask_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('L')
            img.load()  # 显示加载图片
            return img


if __name__ == '__main__':
    pass
