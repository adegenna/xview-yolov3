import numpy as np
import sys
sys.path.append('../')
import scipy.io
import os
import cv2
from utils.fcn_sigma_rejection import *
from utils.datasetProcessing import *

# This is a python-conversion of utils/analysis.m and all related target preprocessing

# ASSUMPTIONS ******************
# 1) Training data filenames have to of form 'string[split]number.extension' or 'number.extension'
# 2) Training datatype is .bmp (this should probably not be hardcoded...)
# ******************************

class Target():
    def __init__(self,inputs):
        self.__inputs             = inputs
        self.__datatype_extension = '.bmp'
        self.load_target_file()
        files, HWC = get_dataset_height_width_channels(self.__inputs.traindir,self.__datatype_extension)
        for i in range(len(self.__chips)):
            self.__chips[i] = strip_image_number_from_filename(self.__chips[i],'_')
        for i in range(len(files)):
            files[i] = strip_image_number_from_filename(files[i],'_')
        self.__files = files.astype('int')
        self.__HWC   = HWC
        self.__x1       = None
        self.__x2       = None
        self.__y1       = None
        self.__y2       = None
        self.__image_h  = None
        self.__image_w  = None
        self.__area     = None
        self.__new_area = None
        self.__w        = None
        self.__h        = None
        self.__new_AR   = None

    def load_target_file(self):
        if (self.__inputs.targetfiletype == 'json'):
            self.__extension = '.json'
            self.__coords, self.__chips, self.__classes = get_labels_geojson(self.__inputs.targetfile)
            self.__class_labels = np.unique(self.__classes)
        else:
           sys.exit('Target file either not specified or not supported')

    def crop(self):
        x1,y1,x2,y2     = parse_xy_coords(self.__coords)
        _,_,self.__area = compute_width_height_area(x1,y1,x2,y2)
        image_w         = np.zeros_like(x1)
        image_h         = np.zeros_like(x1)
        for i in range(len(image_w)):
            idx        = np.where(self.__files == self.__chips[i])[0][0]
            image_h[i] = self.__HWC[idx,0];
            image_w[i] = self.__HWC[idx,1];
        x1 = np.minimum( np.maximum(x1,0), image_w);
        y1 = np.minimum( np.maximum(y1,0), image_h);
        x2 = np.minimum( np.maximum(x2,0), image_w);
        y2 = np.minimum( np.maximum(y2,0), image_h);
        w,h,new_area  = compute_width_height_area(x1,y1,x2,y2)
        new_AR        = np.maximum(w/h, h/w);
        self.__coords = concatenate_xy_to_coords(x1,y1,x2,y2)
        self.__x1       = x1
        self.__x2       = x2
        self.__y1       = y1
        self.__y2       = y2
        self.__image_h  = image_h
        self.__image_w  = image_w
        self.__w        = w
        self.__h        = h
        self.__new_area = new_area
        self.__new_AR   = new_AR

    def sigma_rejection_indices(self):
        i1                  = np.ones_like(self.__x1,dtype='int')
        i2                  = np.ones_like(self.__x1,dtype='int')
        i3                  = np.ones_like(self.__x1,dtype='int')
        for i in range(len(self.__class_labels)):
            idx = np.where(self.__classes == self.__class_labels[i])[0]
            _,v   = fcn_sigma_rejection(self.__new_area[idx],12,3)
            i1[idx] = i1[idx] & v
            _,v   = fcn_sigma_rejection(self.__w[idx],12,3)
            i2[idx] = i2[idx] & v
            _,v   = fcn_sigma_rejection(self.__h[idx],12,3)
            i3[idx] = i3[idx] & v
        return i1,i2,i3;

    def manual_dimension_requirements(self,area_lim,w_lim,h_lim,AR_lim):
        return ( (self.__new_area >= area_lim) & \
                 (self.__w > w_lim) & \
                 (self.__h > h_lim) & \
                 (self.__new_AR < AR_lim) )

    def edge_requirements(self,w_lim,h_lim,x2_lim,y2_lim):
        # Extreme edges (i.e. don't start an x1 10 pixels from the right side)
        return ( (self.__x1 < (self.__image_w-w_lim)) & \
                 (self.__y1 < (self.__image_h-h_lim)) & \
                 (self.__x2 > x2_lim) & \
                 (self.__y2 > y2_lim) )
    
    def clean_coords(self):
        i0         = detect_nans_and_infs_by_row(self.__coords)
        i1,i2,i3   = self.sigma_rejection_indices()
        i4         = self.manual_dimension_requirements(20,4,4,15)
        i5         = self.edge_requirements(self,10,10,10,10)
        i6         = area_requirement(self.__new_area,self.__area,0.25)
        i7         = nan_inf_size_requirements(self.__image_h,self.__image_w,32)
        i8         = invalid_class_requirement(self.__inputs.invalid_class_list,self.__classes,self.__coords)
        valid      = i0 & i1 & i2 & i3 & i4 & i5 & i6 & i7 & i8;
        self.__coords = self.__coords[valid,:];
        
        
    


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

def detect_nans_and_infs_by_row(arr2d):
    assert(len(arr2d.shape) == 2)
    return ~np.any(np.isnan(arr2d) | np.isinf(arr2d) , axis=1)

def area_requirement(new_area,area,area_ratio):
    # Cut objects that lost >90% of their area during crop
    new_area_ratio = new_area / area;
    i6             = new_area_ratio > area_ratio;
    return i6

def nan_inf_size_requirements(image_h,image_w,size):
    # no image dimension nans or infs, or smaller than 32 pix
    hw = np.vstack([image_h, image_w]).T
    i7 = ~np.any( (np.isnan(hw) | np.isinf(hw)) | (hw < size) , axis = 1);
    return i7

def invalid_class_requirement(invalid_class_list,classes,coords):
    # remove invalid classes 75 and 82 (e.g., 'None' class in xview)
    invalid_idx        = np.where( classes == invalid_class_list[:,None] )[1]
    i8                 = np.ones_like(coords,dtype='int')
    i8[invalid_idx]    = 0
    return i8

