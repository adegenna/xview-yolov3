import unittest
import numpy as np
import torch

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


if __name__ == '__main__':
    unittest.main();
