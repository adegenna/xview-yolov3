import sys
sys.path.append('../../')
import glob
import math
import os
import random
import sys
import cv2
import numpy as np
import scipy.io
import torch
from PIL import Image

# from torch.utils.data import Dataset
from utils.utils import xyxy2xywh, xview_class_weights, load_obj, convert_tif2bmp, readBmpDataset, convert_class_labels_to_indices
from src.targets import *
from datasets.datasetTransformations import *
from datasets.datasetStats import *

class ListDataset():  # for training
    """
    Image dataset class for training
    """
    def __init__(self, inputs, targets):
        print('********************* DATA PREPROCESSING *********************')
        print('Reading dataset from ' + inputs.traindir + '...');
        self.path       = inputs.traindir;
        self.files      = readBmpDataset(self.path);
        self.nF         = len(self.files)  # number of image files
        self.nB         = math.ceil(self.nF / inputs.batchsize)  # number of batches
        self.batch_size = inputs.batchsize
        self.height     = inputs.imgsize
        self.targets_path   = inputs.targetspath
        self.targetfiletype = inputs.targetfiletype
        self.__inputs       = inputs
        print('Successfully loaded ' + str(self.nF) + ' images.')
        self.labels   = None
        self.labels1  = None
        self.area0    = None
        self.nL       = None
        self.nL1      = None
        self.r        = None
        # load targets
        if (self.targetfiletype == 'json'):
            print("Loading target data from specified json file...")
            self.targetIDs      = targets.filtered_chips
            coords              = targets.filtered_coords
            classes             = targets.filtered_classes            
            unique_class_labels = targets.list_of_unique_class_labels            
            classes             = convert_class_labels_to_indices(classes,unique_class_labels)
            self.targets        = np.hstack([np.reshape(classes,[len(classes),1]),coords]).astype(float)
            self.targets_metadata = targets
            self.class_weights    = targets.filtered_class_weights
        else:
            sys.exit('Specified target filetype is not supported')
        
        # RGB normalization values
        rgb_mean,rgb_std = compute_dataset_rgb_stats(self.files)
        self.rgb_mean = np.array(rgb_mean , dtype=np.float32).reshape((1, 3, 1, 1))
        self.rgb_std  = np.array(rgb_std  , dtype=np.float32).reshape((1, 3, 1, 1))
        np.savetxt(self.__inputs.outdir + 'training_rgb_mean.out' , rgb_mean , delimiter = ',')
        np.savetxt(self.__inputs.outdir + 'training_rgb_std.out'  , rgb_std  , delimiter = ',')

        print('**************************************************************')
        print('\n');

    def __iter__(self):
        self.count = -1
        if (self.__inputs.sampling_weight == 'inverse_class_frequency'):
            image_labels =  self.targets_metadata.files
            image_weights = self.targets_metadata.image_weights            
            self.shuffled_vector = np.random.choice(image_labels, self.nF, p=image_weights)
        elif (self.__inputs.sampling_weight == 'uniform'):
            self.shuffled_vector = np.random.permutation(self.nF)
        else:
            sys.exit('Specified option sampling_weight is not supported.') 
        return self        
    
    # @profile
    def __next__(self):
        self.count += 1
        if self.count == self.nB:
            raise StopIteration

        ia = self.count * self.batch_size
        ib = min((self.count + 1) * self.batch_size, self.nF)

        height = self.height

        img_all = []
        labels_all = []
        for index, files_index in enumerate(range(ia, ib)):
            img_path = self.__inputs.traindir + str(self.shuffled_vector[files_index]) + '.bmp'

            img0 = cv2.imread(img_path)
            if img0 is None:
                continue

            augment_hsv = False
            if augment_hsv:
                augmentHSV(img0)

            # Load labels
            chip = img_path.rsplit('/')[-1]
            chip = chip.rsplit('_')[-1]
            i = (self.targetIDs == float(chip.replace('.tif', '').replace('.bmp', ''))).nonzero()[0]
            labels1 = self.targets[i]
            img1, labels1, M = random_affine(img0, targets=labels1, degrees=(-20, 20), translate=(0.01, 0.01),
                                             scale=(0.70, 1.30))  # RGB

            self.labels1  = labels1
            self.r        = pickRandomPoints(100,img0,height,M,img1)
            self.nL1      = len(labels1)
            
            if self.nL1 > 0:
                self.pickRandom8Points()
            
            h, w, _ = img1.shape
            for j in range(8):
                self.labels = np.array([], dtype=np.float32)
                pad_x,pad_y = self.eliminateBadLabels(j)
                img = img1[pad_y:pad_y + self.height, pad_x:pad_x + self.height]

                self.nL = len(self.labels)
                if self.nL > 0:
                    # convert labels to xywh
                    self.labels[:, 1:5] = xyxy2xywh(self.labels[:, 1:5].copy()) / self.height

                # random lr flip
                if random.random() > 0.5:
                    img = np.fliplr(img)
                    if self.nL > 0:
                        self.labels[:, 1] = 1 - self.labels[:, 1]

                # random ud flip
                if random.random() > 0.5:
                    img = np.flipud(img)
                    if self.nL > 0:
                        self.labels[:, 2] = 1 - self.labels[:, 2]

                img_all.append(img)
                labels_all.append(torch.from_numpy(self.labels).float())

        # Randomize
        i           = np.random.permutation(len(labels_all))
        img_all     = [img_all[j] for j in i]
        labels_all  = [labels_all[j] for j in i]

        # Normalize
        img_all = np.stack(img_all)[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB and cv2 to pytorch
        img_all = np.ascontiguousarray(img_all, dtype=np.float32)
        img_all -= self.rgb_mean
        img_all /= self.rgb_std

        return torch.from_numpy(img_all), labels_all

    def __len__(self):
        return self.nB  # number of batches

    def pickRandom8Points(self):
        # Random selection of 8 points, weighted by something    
        weights = []
        for k in range(len(self.r)):
            x = (self.labels1[:, 1] + self.labels1[:, 3]) / 2
            y = (self.labels1[:, 2] + self.labels1[:, 4]) / 2
            c = self.labels1[(abs(self.r[k, 0] - x) < self.height / 2) & (abs(self.r[k, 1] - y) < self.height / 2), 0]
            if len(c) == 0:
                weights.append(1e-16)
            else:
                weights.append(self.class_weights[c.astype(np.int8)].sum())
        weights    = np.array(weights)
        weights   /= weights.sum()
        self.r     = self.r[np.random.choice(len(self.r), size=8, p=weights, replace=False)]
        self.area0 = (self.labels1[:, 3] - self.labels1[:, 1]) * (self.labels1[:, 4] - self.labels1[:, 2])

    def eliminateBadLabels(self,j):
        # Eliminate labels that don't satisfy some criteria
        pad_x  = int(self.r[j, 0] - self.height / 2)
        pad_y  = int(self.r[j, 1] - self.height / 2)
        if self.nL1 > 0:
            self.labels = self.labels1.copy()    
            self.labels[:, [1, 3]] -= pad_x
            self.labels[:, [2, 4]] -= pad_y
            np.clip(self.labels[:, 1:5], 0, self.height, out=self.labels[:, 1:5])
            lw = self.labels[:, 3] - self.labels[:, 1]
            lh = self.labels[:, 4] - self.labels[:, 2]
            area = lw * lh
            ar = np.maximum(lw / (lh + 1e-16), lh / (lw + 1e-16))
            # objects must have width and height > 4 pixels
            self.labels = self.labels[(lw > 4) & (lh > 4) & (area > 20) & (area / self.area0 > 0.1) & (ar < 10)]
        return pad_x,pad_y        


