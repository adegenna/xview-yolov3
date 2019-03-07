import glob
import math
import os
import random
import sys
import cv2
import numpy as np
import scipy.io
import torch

# NOTE: hard assumption made that data are .bmp (change this)

class ImageFolder():  # for eval-only
    def __init__(self, inputs):
        path       = inputs.imagepath
        batch_size = inputs.batch_size
        img_size   = inputs.imgsize
        if os.path.isdir(path):
            self.files = sorted(glob.glob('%s/*.bmp*' % path))
        elif os.path.isfile(path):
            self.files = [path]
        self.nF = len(self.files)  # number of image files
        self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.batch_size = batch_size
        self.height = img_size
        assert self.nF > 0, 'No images found in path %s' % path

        # RGB normalization values
        rgb_mean = np.loadtxt(inputs.rgb_mean , delimiter = ',')
        rgb_std  = np.loadtxt(inputs.rgb_std  , delimiter = ',')
        self.rgb_mean = np.array(rgb_mean , dtype=np.float32).reshape((3, 1, 1))
        self.rgb_std  = np.array(rgb_std  , dtype=np.float32).reshape((3, 1, 1))
        
        #self.rgb_mean = np.array([60.134, 49.697, 40.746], dtype=np.float32).reshape((3, 1, 1))
        #self.rgb_std = np.array([29.99, 24.498, 22.046], dtype=np.float32).reshape((3, 1, 1))

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nB:
            raise StopIteration
        img_path = self.files[self.count]

        # Add padding
        img = cv2.imread(img_path)  # BGR

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img -= self.rgb_mean
        img /= self.rgb_std

        return [img_path], img

    def __len__(self):
        return self.nB  # number of batches

