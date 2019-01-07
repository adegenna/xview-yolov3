import sys
sys.path.append('../')
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
from utils.utils import xview_classes2indices

# from torch.utils.data import Dataset
from utils.utils import xyxy2xywh, xview_class_weights, load_obj, convert_tif2bmp, readBmpDataset
from src.targets import *

class ImageFolder():  # for eval-only
    def __init__(self, path, batch_size=1, img_size=416):
        if os.path.isdir(path):
            self.files = sorted(glob.glob('%s/*.*' % path))
        elif os.path.isfile(path):
            self.files = [path]

        self.nF = len(self.files)  # number of image files
        self.nB = math.ceil(self.nF / batch_size)  # number of batches
        self.batch_size = batch_size
        self.height = img_size
        assert self.nF > 0, 'No images found in path %s' % path

        # RGB normalization values
        self.rgb_mean = np.array([60.134, 49.697, 40.746], dtype=np.float32).reshape((3, 1, 1))
        self.rgb_std = np.array([29.99, 24.498, 22.046], dtype=np.float32).reshape((3, 1, 1))

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


class ListDataset():  # for training
    def __init__(self, inputs):
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
        print('Successfully loaded ' + str(self.nF) + ' images.')
        self.labels   = None
        self.labels1  = None
        self.area0    = None
        self.nL       = None
        self.nL1      = None
        self.r        = None
        # load targets
        if (self.targetfiletype == 'matlab'):
            print("Loading target data from specified matlab file...")
            self.mat = scipy.io.loadmat(self.targets_path)
            self.targetIDs     = self.mat['id'].squeeze()
            self.targets       = self.mat['targets']
        elif (self.targetfiletype == 'pickle'):
            print("Loading target data from specified pickle file...")
            self.mat = load_obj(self.targets_path)
            self.targetIDs     = self.mat['id'].squeeze()
            self.targets       = self.mat['targets']
        elif (self.targetfiletype == 'json'):
            print("Loading target data from specified json file...")
            targets         = Target(inputs)
            self.targetIDs  = vars(targets)['_Target__chips']
            coords          = vars(targets)['_Target__filtered_coords']
            classes         = vars(targets)['_Target__filtered_classes']
            classes         = xview_classes2indices(classes)
            self.targets    = np.hstack([np.reshape(classes,[len(classes),1]),coords])
            self.targets_metadata = targets
        else:
            sys.exit('Specified target filetype is not supported')
        
        self.class_weights = xview_class_weights(range(60)).numpy()
        # RGB normalization values
        self.rgb_mean = np.array([60.134, 49.697, 40.746], dtype=np.float32).reshape((1, 3, 1, 1))
        self.rgb_std = np.array([29.99, 24.498, 22.046], dtype=np.float32).reshape((1, 3, 1, 1))
        # RGB normalization of HSV-equalized images
        # self.rgb_mean = np.array([122.367, 107.586, 86.987], dtype=np.float32).reshape((1, 3, 1, 1))
        # self.rgb_std = np.array([65.914, 55.797, 47.340], dtype=np.float32).reshape((1, 3, 1, 1))

        # RGB normalization of YUV-equalized images clipped at 5
        # self.rgb_mean = np.array([100.931, 90.863, 82.412], dtype=np.float32).reshape((1, 3, 1, 1))
        # self.rgb_std = np.array([52.022, 47.313, 44.845], dtype=np.float32).reshape((1, 3, 1, 1))

        # RGB normalization of YUV-equalized images clipped at 3
        # self.rgb_mean = np.array([45.068, 40.035, 37.538], dtype=np.float32).reshape((1, 3, 1, 1))
        # self.rgb_std = np.array([89.836, 79.490, 71.011], dtype=np.float32).reshape((1, 3, 1, 1))

        # RGB normalization of YUV-equalized images no clipping
        # self.rgb_mean = np.array([137.513, 127.813, 119.410], dtype=np.float32).reshape((1, 3, 1, 1))
        # self.rgb_std = np.array([69.095, 66.369, 64.236], dtype=np.float32).reshape((1, 3, 1, 1))

        print('**************************************************************')
        print('\n');

    def __iter__(self):
        self.count = -1
        self.shuffled_vector = np.random.permutation(self.nF)  # shuffled vector
        #self.shuffled_vector = np.random.choice(self.mat['image_numbers'].ravel(), self.nF,
        #                                        p=self.mat['image_weights'].ravel())
        return self        
    
    # @profile
    def __next__(self):
        self.count += 1
        if self.count == self.nB:
            raise StopIteration

        ia = self.count * self.batch_size
        ib = min((self.count + 1) * self.batch_size, self.nF)

        height = self.height
        # height = random.choice([15, 17, 19, 21]) * 32

        img_all = []
        labels_all = []
        for index, files_index in enumerate(range(ia, ib)):
            # img_path = self.files[self.shuffled_vector[files_index]]  # BGR
            # img_path = '%s%g.bmp' % (self.path, self.shuffled_vector[files_index])
            img_path = self.files[self.shuffled_vector[files_index]]

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
            
            # Remove buildings and small cars
            # labels1 = labels1[(labels1[:, 0] != 5) & (labels1[:, 0] != 48)]

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
                    # remap xview classes 11-94 to 0-61
                    # labels[:, 0] = xview_classes2indices(labels[:, 0])

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
                labels_all.append(torch.from_numpy(self.labels))

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
                print(c)
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



def pickRandomPoints(pts,img0,height,M,img1):
    # Pick random points inside image
    border = height / 2 + 1
    r = np.ones((pts, 3))
    r[:, :2] = np.random.rand(pts, 2) * (np.array(img0.shape)[[1, 0]] - border * 2) + border
    r = (r @ M.T)[:, :2]
    r = r[np.all(r > border, 1) & np.all(r < img1.shape[0] - border, 1)]
    return r


def augmentHSV(img0):
    # SV augmentation by 50%
    fraction = 0.50
    img_hsv = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
    S = img_hsv[:, :, 1].astype(np.float32)
    V = img_hsv[:, :, 2].astype(np.float32)
    a = (random.random() * 2 - 1) * fraction + 1
    S *= a
    if a > 1:
        np.clip(S, a_min=0, a_max=255, out=S)
    a = (random.random() * 2 - 1) * fraction + 1
    V *= a
    if a > 1:
        np.clip(V, a_min=0, a_max=255, out=V)
    img_hsv[:, :, 1] = S.astype(np.uint8)
    img_hsv[:, :, 2] = V.astype(np.uint8)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img0)


def resize_square(img, height=416, color=(0, 0, 0)):  # resizes a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)
    new_shape = [round(shape[0] * ratio), round(shape[1] * ratio)]
    dw = height - new_shape[1]  # width padding
    dh = height - new_shape[0]  # height padding
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    img = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_AREA)
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def random_affine(img, targets=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-3, 3),
                  borderValue=(0, 0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    border = 750
    height = max(img.shape[0], img.shape[1]) + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    a += random.choice([-180, -90, 0, 90])  # random 90deg rotations added to small rotations

    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(height, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order (YUV-equalized BGR means)
    # borderValue = [40.746, 49.697, 60.134])  # RGB

    # Return warped points also
    if targets is not None:
        if len(targets) > 0:
            n = targets.shape[0]
            points = targets[:, 1:5].copy()
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            np.clip(xy, 0, height, out=xy)
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / area0 > 0.1) & (ar < 10)

            targets = targets[i]
            targets[:, 1:5] = xy[i]

        return imw, targets, M
    else:
        return imw



