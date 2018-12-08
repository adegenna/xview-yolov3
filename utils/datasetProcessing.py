import numpy as np
import sys
sys.path.append('../')
import scipy.io
import os
import cv2
import json

# Simple functions commonly used for (pre-)processing of datasets

def get_dataset_height_width_channels(datadir,extension):
    files  = get_dataset_filenames(datadir,extension)
    nfiles = len(files)
    HWC    = np.zeros([nfiles,3])
    for i in range(nfiles):
        img    = cv2.imread(datadir + files[i])
        h,w,c  = img.shape
        HWC[i,0] = h;
        HWC[i,1] = w;
        HWC[i,2] = c;
    return files, HWC
        
def get_dataset_filenames(datadir,extension):
    files    = []
    for file in os.listdir(datadir):
        if file.endswith(extension):
            datum = file
            files = np.append(files,datum)
    return files

def get_labels_geojson(fname="xView_train.geojson"):
    # Processes an xView GeoJSON file
    # INPUT: filepath to the GeoJSON file
    # OUTPUT: Bounding box coordinate array, Chip-name array, and Class-id array
    with open(fname) as f:
        data = json.load(f)
    coords = np.zeros((len(data['features']),4))
    chips = np.zeros((len(data['features'])),dtype="object")
    classes = np.zeros((len(data['features'])))
    for i in range(len(data['features'])):
        if data['features'][i]['properties']['bounds_imcoords'] != []:
            b_id = data['features'][i]['properties']['image_id']
            val = np.array([int(num) for num in data['features'][i]['properties']['bounds_imcoords'].split(",")])
            chips[i] = b_id
            classes[i] = data['features'][i]['properties']['type_id']
            coords[i] = val
        else:
            chips[i] = 'None'
    return coords, chips, classes

def strip_image_number_from_filename(imgname,splitchar):
    # Strips away file extension AND strings prepended to image number that are separated with splitchar
    # e.g., train_image_1234.tif --> 1234
    num = imgname.split(splitchar)[-1]
    num = int(num.split('.')[0])
    return num

def strip_geojson(data,ids):
    # Strip away all data from geojson file except those images whose numbers are specified by ids
    datafeatureskeep = []
    for i in range(len(ids)):
        datafeaturesI    = [d for d in data['features'] if d['properties']['image_id'] == str(ids[i]) + '.tif']
        datafeatureskeep = np.append(datafeatureskeep,datafeaturesI)
    data['features'] = list(datafeatureskeep)
    return data;
