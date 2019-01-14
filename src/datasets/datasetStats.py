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

def compute_dataset_rgb_stats(dataloader_files):
    """
    Function to compute rgb statistics of a given dataset. Uses a two-pass sequential algorithm.

    | **Inputs:**
    |    *dataloader_files:* list of absolute paths of all dataset files
    
    | **Outputs:**
    |    *(mean_rgb,std_rgb):* mean and standard deviation of RGB channels of dataset
    """
    print("Computing dataset image RGB statistics")
    nfiles      = len(dataloader_files)
    npixels     = 0;
    channelsum  = np.zeros(3)
    channelsum_var = np.zeros(3)
    # First pass: mean
    for i in range(nfiles):
        img_path    = dataloader_files[i]
        img         = cv2.imread(img_path)
        channelsum += np.sum(img,axis=(0,1))
        npixels    += np.prod(img.shape[0:2])
    mean_rgb  = channelsum/npixels
    # Second pass: stddev
    for i in range(nfiles):
        img_path    = dataloader_files[i]
        img             = cv2.imread(img_path)
        channelsum_var += np.sum( (img - mean_rgb)**2 , axis=(0,1))
    std_rgb   = np.sqrt(channelsum_var/npixels)
    mean_rgb  = np.flip(mean_rgb)
    std_rgb   = np.flip(std_rgb)
    return mean_rgb,std_rgb

def compute_dataset_rgb_stats_load_into_memory(dataloader_files):
    """
    Function to compute rgb statistics of a given dataset. Loads all data into memory and computes statistics in one-shot.
    
    | **Inputs:**
    |    *dataloader_files:* list of absolute paths of all dataset files
    
    | **Outputs:**
    |    *(mean_rgb,std_rgb):* mean and standard deviation of RGB channels of dataset

    """
    print("Computing dataset image RGB statistics")
    nfiles      = len(dataloader_files)
    data        = np.zeros(3)
    for i in range(nfiles):
        img_path    = dataloader_files[i]
        img         = cv2.imread(img_path)
        bchan  = np.ravel(img[:,:,0])
        gchan  = np.ravel(img[:,:,1])
        rchan  = np.ravel(img[:,:,2])
        data_i = np.zeros([np.prod(img.shape[0:2]),3])
        data_i[:,0] = bchan
        data_i[:,1] = gchan
        data_i[:,2] = rchan
        if (i==0):
            data = data_i
        else:
            data        = np.vstack([data,data_i])
    mean_rgb = np.flip(np.mean(data,axis=0))
    std_rgb  = np.flip(np.std(data,axis=0))
    return mean_rgb,std_rgb
