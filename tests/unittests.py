import unittest
import numpy as np
import torch
import sys,os
import matplotlib.pyplot as plt
import cv2
import json
sys.path.append('../')
from utils.datasets import *
from src.InputFile import *
from utils.utils import plot_rgb_image
from utils.datasetProcessing import *
import warnings

class GPUtests(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore",category=ResourceWarning)
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        
    def test_cuda_available(self):
        print("test if cuda is available")
        cudaavail = torch.cuda.is_available()
        print("cuda available: " + str(cudaavail))
        self.assertTrue(cudaavail);
    def test_cuda_version(self):
        print("test if cuda version is > 9")
        cudarelease = torch.version.cuda
        print("cuda version: " + str(cudarelease))
        self.assertTrue(int(cudarelease[0]) >= 9);
    def test_gpu_avail(self):
        print("test if GPUs are available")
        numGPU = torch.cuda.device_count()
        print("gpus available: " + str(numGPU));
        for i in range(numGPU):
            namegpu = torch.cuda.get_device_name(i)
            print("gpu " + str(i+1) + ": " + namegpu);
        self.assertTrue(numGPU >= 1);


        
class DataProcessingTests(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore",category=ResourceWarning)
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        args                    = lambda:0
        args.inputfilename      = './input_test.dat'
        self.inputs             = InputFile(args)
        self.ndata              = 9
        self.nobject            = 11158

    def test_get_labels_geojson(self):
        print('test loading of geojson formatted data')
        coords,chips,classes    = get_labels_geojson(self.inputs.datadir + 'xview/labels/jsontest.json')
        self.assertTrue(coords.shape == (self.nobject, 4))
        self.assertTrue(chips.size   == self.nobject)
        self.assertTrue(classes.size == self.nobject)

    def test_get_dataset_filenames(self):
        print('test loading dataset filenames')
        files  = get_dataset_filenames(self.inputs.traindir,'.tif')
        self.assertTrue(len(files) == self.ndata)

    def test_get_dataset_height_width_channels(self):
        print('test loading sizes of dataset images')
        extension  = '.tif'
        files, HWC = get_dataset_height_width_channels(self.inputs.traindir,extension)
        self.assertTrue(HWC.shape == (self.ndata,3))
        self.assertTrue(files[0]  == ('train_images_2316' + extension))
        self.assertTrue(np.all(HWC[0] == np.array([3197, 3475, 3])))

    def test_strip_image_number_from_filename(self):
        print('test strip image number from image filename')
        imgname  = 'train_images_2316.tif'
        imgname2 = '2316.tif'
        num      = strip_image_number_from_filename(imgname,'_')
        num2     = strip_image_number_from_filename(imgname2,'_')
        self.assertTrue(num  == 2316)
        self.assertTrue(num2 == 2316)
        

        
class DatasetTests(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore",category=ResourceWarning)
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        args                    = lambda:0
        args.inputfilename      = './input_test.dat'
        self.inputs             = InputFile(args);
        self.filetypes          = ['matlab', 'pickle']
        self.filetypeAppend     = ['.mat', '.pkl']
        self.basetargetpath     = self.inputs.targetspath;
        self.nclass             = 60
        self.nobjects           = 11057
        self.ndata              = 9
        self.expectedTargetKeys = {'__header__', '__version__', '__globals__', 'class_cov', 'class_mu', 'class_sigma', 'id', 'image_numbers', 'image_weights', 'targets', 'wh'}
    
    def test_load_targets(self):
        print('test training data loading functionality')        
        for i in range(len(self.filetypes)):
            self.inputs.targetfiletype = self.filetypes[i]
            self.inputs.targetspath    = self.basetargetpath + self.filetypeAppend[i];
            sys.stdout = open(os.devnull, 'w')
            dataloader                 = ListDataset(self.inputs)
            sys.stdout = sys.__stdout__
            self.assertTrue(dataloader.mat.keys() == self.expectedTargetKeys)
            self.assertTrue(dataloader.mat['class_cov'].shape   == (self.nclass,4,4))
            self.assertTrue(dataloader.mat['class_mu'].shape    == (self.nclass,4))
            self.assertTrue(dataloader.mat['class_sigma'].shape == (self.nclass,4))
            self.assertTrue(dataloader.mat['id'].size              == self.nobjects)
            self.assertTrue(dataloader.mat['image_numbers'].shape  == (self.ndata,1))
            self.assertTrue(dataloader.mat['image_weights'].shape  == (self.ndata,1))
            self.assertTrue(dataloader.mat['targets'].shape        == (self.nobjects,5))
            self.assertTrue(dataloader.mat['wh'].shape             == (self.nobjects,2))
            
    def test_show_targets(self):
        print('test training data labeling')
        self.inputs.targetfiletype = self.filetypes[0]
        self.inputs.targetspath    = self.basetargetpath + self.filetypeAppend[0];
        sys.stdout = open(os.devnull, 'w')
        dataloader                 = ListDataset(self.inputs)
        sys.stdout = sys.__stdout__
        for i, (imgs, targets) in enumerate(dataloader):
            try:
                obj     = targets[0][:,1:].numpy()
                obj    *= self.inputs.imgsize
            except:
                obj     = []
            img     = np.transpose(imgs[0].numpy(),(1,2,0))
            plot_rgb_image(img,dataloader.rgb_mean.squeeze(),dataloader.rgb_std.squeeze(),obj)

            
if __name__ == '__main__':
    unittest.main();
