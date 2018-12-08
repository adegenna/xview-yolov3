import numpy as np
import sys
sys.path.append('../')
import scipy.io
import os
import cv2

from utils.datasetProcessing import *

# This is a python-conversion of utils/analysis.m and all related target preprocessing

# ASSUMPTIONS ******************
# 1) Training data filenames have to of form 'string[split]number.extension' or 'number.extension'
# ******************************

class Target():
    def __init__(self,inputs):
        self.__inputs = inputs
        self.load_target_file()
        files, HWC = get_dataset_height_width_channels(self.__inputs.datadir,self.__extension)
        for i in range(len(self.__chips)):
            self.__chips[i] = strip_image_number_from_filename(self.__chips[i],'_')
        for i in range(len(files)):
            files[i] = strip_image_number_from_filename(files[i],'_')
        self.__files = files
        self.__HWC   = HWC
            
    def load_target_file(self):
        if (self.__inputs.targetfiletype == 'json'):
            self.__extension = '.json'
            self.__coords, self.__chips, self.__classes = get_labels_geojson(self.__inputs.targetfile)
        else:
            sys.exit('Target file either not specified or not supported')

    def crop(self):
        x1,y1,x2,y2 = parse_xy_coords(self.__coords)
        image_w     = np.zeros_like(x1)
        image_h     = np.zeros_like(x1)
        for i in range(len(image_w)):
            idx        = np.where(self.__files == self.__chips[i])[0][0]
            image_h[i] = HWC[idx,0]; image_w[i] = HWC[idx,1];
        x1 = min( max(x1,0), image_w);
        y1 = min( max(y1,0), image_h);
        x2 = min( max(x2,0), image_w);
        y2 = min( max(y2,0), image_h);
        w,h,new_area  = compute_width_height_area(x1,y1,x2,y2)
        new_ar        = max(w/h, h/w);
        self.__coords = concatenate_xy_to_coords(x1,y1,x2,y2)
    
    def clean_coords(self):
        xmin,ymin,xmax,ymax = parse_xy_coords(self.__coords)
        w,h,area            = compute_width_height_area(xmin,ymin,xmax,ymax)
        self.crop()
        
        
    


# Little auxiliary functions
def parse_xy_coords(coords):
    xmin = coords[:,0]
    ymin = coords[:,1]
    xmax = coords[:,2]
    ymax = coords[:,3]
    return xmin,ymin,xmax,ymax

def concatenate_xy_to_coords(xmin,ymin,xmax,ymax):
    coords = np.vstack([xmin,ymin,xmax,ymax]).T
    return coords

def compute_width_height_area(xmin,ymin,xmax,ymax):
    w = xmax-xmin
    h = ymax-ymin
    area = w*h
    return w,h,area    










