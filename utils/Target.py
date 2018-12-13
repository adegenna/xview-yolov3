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
        self.__files, self.__HWC  = get_dataset_height_width_channels(self.__inputs.traindir,self.__datatype_extension)
        self.strip_image_number_from_chips_and_files()
        self.__x1,self.__y1,self.__x2,self.__y2  = parse_xy_coords(self.__coords)
        self.__w,self.__h,self.__area            = compute_width_height_area(self.__x1,self.__y1,self.__x2,self.__y2)
        self.set_image_w_and_h()
        self.__mask              = None
        self.__filtered_x1       = None
        self.__filtered_y1       = None
        self.__filtered_x2       = None
        self.__filtered_y2       = None
        self.__filtered_w        = None
        self.__filtered_h        = None
        self.__filtered_area     = None 
        self.__filtered_AR       = None
        self.__filtered_coords   = None
        self.__filtered_chips    = None
        self.__filtered_classes  = None
        self.__filtered_image_h  = None
        self.__filtered_image_w  = None
        self.__filtered_files    = None

    def set_image_w_and_h(self):
        self.__image_w         = np.zeros_like(self.__x1)
        self.__image_h         = np.zeros_like(self.__x1)
        for i in range(len(self.__image_w)):
            idx               = np.where(self.__files == self.__chips[i])[0][0]
            self.__image_h[i] = self.__HWC[idx,0];
            self.__image_w[i] = self.__HWC[idx,1];

    def strip_image_number_from_chips_and_files(self):
        for i in range(len(self.__chips)):
            self.__chips[i] = strip_image_number_from_filename(self.__chips[i],'_')
        for i in range(len(self.__files)):
            self.__files[i] = strip_image_number_from_filename(self.__files[i],'_')
        self.__files    = self.__files.astype('int')
            
    def load_target_file(self):
        if (self.__inputs.targetfiletype == 'json'):
            self.__extension = '.json'
            self.__coords, self.__chips, self.__classes = get_labels_geojson(self.__inputs.targetfile)
            self.__class_labels = np.unique(self.__classes)
        else:
            sys.exit('Target file either not specified or not supported')

    def compute_cropped_data(self):
        self.__filtered_x1 = np.minimum( np.maximum(self.__x1,0), self.__image_w);
        self.__filtered_y1 = np.minimum( np.maximum(self.__y1,0), self.__image_h);
        self.__filtered_x2 = np.minimum( np.maximum(self.__x2,0), self.__image_w);
        self.__filtered_y2 = np.minimum( np.maximum(self.__y2,0), self.__image_h);
        self.compute_filtered_variables_from_filtered_xy()

    def sigma_rejection_indices(self,filtered_data):
        mask_reject   = np.ones_like(self.__filtered_x1,dtype='int')
        for i in range(len(self.__class_labels)):
            idx = np.where(self.__classes == self.__class_labels[i])[0]
            _,v   = fcn_sigma_rejection(filtered_data[idx],12,3)
            mask_reject[idx] = mask_reject[idx] & v
        return mask_reject

    def manual_dimension_requirements(self,area_lim,w_lim,h_lim,AR_lim):
        return ( (self.__filtered_area >= area_lim) & \
                 (self.__filtered_w > w_lim) & \
                 (self.__filtered_h > h_lim) & \
                 (self.__filtered_AR < AR_lim) )

    def edge_requirements(self,w_lim,h_lim,x2_lim,y2_lim):
        # Extreme edges (i.e. don't start an x1 10 pixels from the right side)
        return ( (self.__filtered_x1 < (self.__image_w-w_lim)) & \
                 (self.__filtered_y1 < (self.__image_h-h_lim)) & \
                 (self.__filtered_x2 > x2_lim) & \
                 (self.__filtered_y2 > y2_lim) )
    
    def compute_filtered_data_mask(self):
        self.compute_cropped_data()
        i0         = detect_nans_and_infs_by_row(self.__coords)
        i1         = self.sigma_rejection_indices(self.__filtered_area)
        i2         = self.sigma_rejection_indices(self.__filtered_w)
        i3         = self.sigma_rejection_indices(self.__filtered_h)
        i4         = self.manual_dimension_requirements(20,4,4,15)
        i5         = self.edge_requirements(10,10,10,10)
        i6         = area_requirement(self.__filtered_area,self.__area,0.25)
        i7         = nan_inf_size_requirements(self.__image_h,self.__image_w,32)
        i8         = invalid_class_requirement(self.__inputs.invalid_class_list,self.__classes)
        valid      = i0 & i1 & i2 & i3 & i4 & i5 & i6 & i7 & i8;
        self.__mask = valid

    def apply_mask_to_filtered_data(self):
        assert(self.__mask != None)
        self.__filtered_coords   = self.__filtered_coords[self.__mask];
        self.__filtered_chips    = self.__chips[self.__mask]
        self.__filtered_classes  = self.__classes[self.__mask]
        self.__filtered_image_h  = self.__image_h[self.__mask]
        self.__filtered_image_w  = self.__image_w[self.__mask]
        self.__filtered_files    = self.__files[self.__mask]
        self.compute_filtered_variables_from_filtered_coords()

    def compute_filtered_variables_from_filtered_coords(self):
        self.__filtered_x1,self.__filtered_y1,self.__filtered_x2,self.__filtered_y2  = parse_xy_coords(self.__filtered_coords)
        self.__filtered_w,self.__filtered_h,self.__filtered_area  = \
                compute_width_height_area(self.__filtered_x1,self.__filtered_y1,self.__filtered_x2,self.__filtered_y2)
        self.__filtered_AR            = np.maximum(self.__filtered_w/self.__filtered_h, self.__filtered_h/self.__filtered_w);

    def compute_filtered_variables_from_filtered_xy(self):
        self.__filtered_w,self.__filtered_h,self.__filtered_area  = \
                compute_width_height_area(self.__filtered_x1,self.__filtered_y1,self.__filtered_x2,self.__filtered_y2)
        self.__filtered_AR            = np.maximum(self.__filtered_w/self.__filtered_h, self.__filtered_h/self.__filtered_w);
        self.__filtered_coords        = concatenate_xy_to_coords(self.__filtered_x1,self.__filtered_y1,self.__filtered_x2,self.__filtered_y2)
        


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

def area_requirement(crop_area,area,area_ratio):
    # Cut objects that lost >90% of their area during crop
    crop_area_ratio = crop_area / area;
    i6              = crop_area_ratio > area_ratio;
    return i6

def nan_inf_size_requirements(image_h,image_w,size):
    # no image dimension nans or infs, or smaller than 32 pix
    hw = np.vstack([image_h, image_w]).T
    i7 = ~np.any( (np.isnan(hw) | np.isinf(hw)) | (hw < size) , axis = 1);
    return i7

def invalid_class_requirement(invalid_class_list,classes):
    # remove invalid classes 75 and 82 (e.g., 'None' class in xview)
    invalid_idx        = np.where( classes == invalid_class_list[:,None] )[1]
    i8                 = np.ones_like(classes,dtype='int')
    i8[invalid_idx]    = 0
    return i8

