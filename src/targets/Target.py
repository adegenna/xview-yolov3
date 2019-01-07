import numpy as np
import sys
sys.path.append('../../')
import scipy.io
import os
import cv2
from sklearn.cluster import KMeans
from targets.fcn_sigma_rejection import *
from targets.per_class_stats import *
from utils.datasetProcessing import *
from utils.utils import convert_tif2bmp, readBmpDataset

# This is a python-conversion of utils/analysis.m and all related target preprocessing

# ****************** ASSUMPTIONS ******************
# 1) Training data filenames have to be of form 'string[split]number.extension' or 'number.extension'
# 2) Training datatype is .bmp (this should probably not be hardcoded...)
# *************************************************

class Target():
    """
    Class for handling target pre-processing tasks.
    """

    def __init__(self,inputs):
        """
        Class constructor. Performs all target processing in Target::process_target_data()

        | **Inputs:**
        |    *inputs:* input file formatted according to InputFile class
        """
        # Start by converting traindir/ dataset to .bmp (if necessary)
        _ = readBmpDataset(inputs.traindir);
        self.__inputs             = inputs
        self.__datatype_extension = '.bmp'
        self.load_target_file()
        self.__files, self.__HWC  = get_dataset_height_width_channels(self.__inputs.traindir,self.__datatype_extension)
        self.strip_image_number_from_chips_and_files()
        self.__x1,self.__y1,self.__x2,self.__y2  = parse_xy_coords(self.__coords)
        self.__w,self.__h,self.__area            = compute_width_height_area(self.__x1,self.__y1,self.__x2,self.__y2)
        self.set_image_w_and_h()
        self.process_target_data()
        
    def set_image_w_and_h(self):
        """
        Method to set width and height of images associated with targets.
        """
        self.__image_w         = np.zeros_like(self.__x1)
        self.__image_h         = np.zeros_like(self.__x1)
        for i in range(len(self.__image_w)):
            try:
                idx           = np.where(self.__files == self.__chips[i])[0][0]
            except:
                print( np.unique(self.__chips) )
                sys.exit('chips_i not found in files list')
            self.__image_h[i] = self.__HWC[idx,0];
            self.__image_w[i] = self.__HWC[idx,1];

    def strip_image_number_from_chips_and_files(self):
        """
        Method to strip numbers from image filenames from both chips and files.
        """
        for i in range(len(self.__chips)):
            self.__chips[i] = strip_image_number_from_filename(self.__chips[i],'_')
        for i in range(len(self.__files)):
            self.__files[i] = strip_image_number_from_filename(self.__files[i],'_')
        self.__files    = self.__files.astype('int')
            
    def load_target_file(self):
        """
        Method to load a targetfile of type specified in the input file. Supported types: .json.
        """
        if (self.__inputs.targetfiletype == 'json'):
            self.__extension = '.json'
            self.__coords, self.__chips, self.__classes = get_labels_geojson(self.__inputs.targetspath)
            self.__class_labels = np.unique(self.__classes)
        else:
            sys.exit('Target file either not specified or not supported')

    def compute_cropped_data(self):
        """
        Method to crop image data based on the width and height. Filtered variables are then computed based on the updated image coordinates.
        """
        self.__filtered_x1 = np.minimum( np.maximum(self.__x1,0), self.__image_w);
        self.__filtered_y1 = np.minimum( np.maximum(self.__y1,0), self.__image_h);
        self.__filtered_x2 = np.minimum( np.maximum(self.__x2,0), self.__image_w);
        self.__filtered_y2 = np.minimum( np.maximum(self.__y2,0), self.__image_h);
        self.compute_filtered_variables_from_filtered_xy()

    def sigma_rejection_indices(self,filtered_data):
        """
        Method to compute a mask based on a sigma rejection criterion.
        
        | **Inputs:** 
        |    *filtered_data:* data to which sigma rejection is applied and from which mask is computed
        
        | **Outputs:**
        |    *mask_reject:* binary mask computed from sigma rejection
        """
        mask_reject   = np.ones_like(self.__filtered_x1,dtype='int')
        for i in range(len(self.__class_labels)):
            idx = np.where(self.__classes == self.__class_labels[i])[0]
            _,v   = fcn_sigma_rejection(filtered_data[idx],12,3)
            mask_reject[idx] = mask_reject[idx] & v
        return mask_reject

    def manual_dimension_requirements(self,area_lim,w_lim,h_lim,AR_lim):
        """
        Method to compute filtering based on specified dimension requirements.
        
        | **Inputs:** 
        |    *area_lim:* limit for image area
        |    *w_lim:* limit for image width
        |    *h_lim:* limit for image height
        |    *AR_lim:* limit for image aspect ratio
        
        | **Outputs:**
        |    indices where filtered variables satisfy the dimension requirements.
        """
        return ( (self.__filtered_area >= area_lim) & \
                 (self.__filtered_w > w_lim) & \
                 (self.__filtered_h > h_lim) & \
                 (self.__filtered_AR < AR_lim) )

    def edge_requirements(self,w_lim,h_lim,x2_lim,y2_lim):
        """
        Method to compute filtering based on edge specifications.
        
        | **Inputs:** 
        |    *w_lim:* limit for image width
        |    *h_lim:* limit for image height
        |    *x2_lim:* limit for image x2
        |    *y2_lim:* limit for image y2
        
        | **Outputs:**
        |    indices where filtered variables satisfy the dimension requirements.
        """
        # Extreme edges (i.e. don't start an x1 10 pixels from the right side)
        return ( (self.__filtered_x1 < (self.__image_w-w_lim)) & \
                 (self.__filtered_y1 < (self.__image_h-h_lim)) & \
                 (self.__filtered_x2 > x2_lim) & \
                 (self.__filtered_y2 > y2_lim) )
    
    def compute_filtered_data_mask(self):
        """
        Method to compute filtered data by applying several filtering operations.
        """
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
        self.__mask = valid.astype(bool)

    def apply_mask_to_filtered_data(self):
        """
        Method to apply mask to filtered data variables.
        """
        try:
            assert(self.__mask is not None)
        except AssertionError as e:
            e.args += ('Filtered data elements must be computed prior to using this function',)
            raise
        self.__filtered_coords       = self.__filtered_coords[self.__mask];
        self.__filtered_chips        = self.__chips[self.__mask]
        self.__filtered_classes      = self.__classes[self.__mask]
        self.__filtered_image_h      = self.__image_h[self.__mask]
        self.__filtered_image_w      = self.__image_w[self.__mask]
        self.__filtered_class_labels = np.unique(self.__filtered_classes)
        self.compute_filtered_variables_from_filtered_coords()

    def compute_filtered_variables_from_filtered_coords(self):
        """
        Method to compute filtered variables from filtered coordinates.
        """
        self.__filtered_x1,self.__filtered_y1,self.__filtered_x2,self.__filtered_y2  = parse_xy_coords(self.__filtered_coords)
        self.__filtered_w,self.__filtered_h,self.__filtered_area  = \
                compute_width_height_area(self.__filtered_x1,self.__filtered_y1,self.__filtered_x2,self.__filtered_y2)
        self.__filtered_AR            = np.maximum(self.__filtered_w/self.__filtered_h, self.__filtered_h/self.__filtered_w);

    def compute_filtered_variables_from_filtered_xy(self):
        """
        Method to compute filtered variables from filtered xy.
        """
        self.__filtered_w,self.__filtered_h,self.__filtered_area  = \
                compute_width_height_area(self.__filtered_x1,self.__filtered_y1,self.__filtered_x2,self.__filtered_y2)
        self.__filtered_AR            = np.maximum(self.__filtered_w/self.__filtered_h, self.__filtered_h/self.__filtered_w);
        self.__filtered_coords        = concatenate_xy_to_coords(self.__filtered_x1,self.__filtered_y1,self.__filtered_x2,self.__filtered_y2)

    def compute_image_weights_with_filtered_data(self):
        """
        Method to compute image weights from filtered data. Weight is simply inverse of class frequency.
        """
        try:
            assert(self.__filtered_classes is not None)
        except AssertionError as e:
            e.args += ('Filtered data elements must be computed prior to using this function',)
            raise
        class_mu, class_sigma, class_cov = per_class_stats(self.__filtered_classes,self.__filtered_w,self.__filtered_h)
        num_class_objects = np.unique(self.__filtered_classes,return_counts=True)[1]
        weights           = 1./num_class_objects
        weights          /= np.sum(weights)
        self.__filtered_class_freq    = num_class_objects
        self.__filtered_class_weights = weights

    def compute_bounding_box_clusters_using_kmeans(self,n_clusters):
        """
        | Method to compute bounding box clusters using kmeans.
        
        | **Inputs:**
        |    *n_clusters:* number of desired kmeans clusters
        """
        HW                 = np.vstack([self.__filtered_w,self.__filtered_h]).T
        kmeans_wh          = KMeans(n_clusters,random_state=0).fit(HW)
        clusters_wh        = kmeans_wh.cluster_centers_
        idx                = np.argsort(clusters_wh[:,0]*clusters_wh[:,1])
        self.__clusters_wh = clusters_wh[idx]

    def process_target_data(self):
        """
        Method to perform all target processing.
        """
        self.compute_cropped_data()
        self.compute_filtered_data_mask()
        self.apply_mask_to_filtered_data()
        self.compute_image_weights_with_filtered_data()
        self.compute_bounding_box_clusters_using_kmeans(self.__inputs.boundingboxclusters)


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

