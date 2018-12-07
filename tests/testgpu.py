import unittest
import numpy as np
import torch
import sys
sys.path.append('../')
from utils.datasets import *
from src.InputFile import *
import matplotlib.pyplot as plt
import cv2
from utils.utils import plot_rgb_image

class GPUtests(unittest.TestCase):
    def test_cuda_available(self):
        cudaavail = torch.cuda.is_available()
        print("\n test if cuda is available")
        print("cuda available: " + str(cudaavail))
        self.assertTrue(cudaavail);
    def test_cuda_version(self):
        print("\n test if cuda version is > 9")
        cudarelease = torch.version.cuda
        print("cuda version: " + str(cudarelease))
        self.assertTrue(int(cudarelease[0]) >= 9);
    def test_gpu_avail(self):
        print("\n test if GPUs are available")
        numGPU = torch.cuda.device_count()
        print("gpus available: " + str(numGPU));
        for i in range(numGPU):
            namegpu = torch.cuda.get_device_name(i)
            print("gpu " + str(i+1) + ": " + namegpu);
        self.assertTrue(numGPU >= 1);

class DatasetTests(unittest.TestCase):
    def setUp(self):
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
            dataloader                 = ListDataset(self.inputs)
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
        dataloader                 = ListDataset(self.inputs)
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

