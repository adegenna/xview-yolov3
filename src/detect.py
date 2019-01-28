import argparse
import time
from sys import platform

import torch
from models import *
from datasets import *
from utils.utils import *
from InputFile import *
from NetworkTester import *

def detect():
    """
    Main driver script for testing the YOLOv3 network.

    | **Inputs:**
    |    *args:* command line arguments used in shell call for this main driver script. args must have a inputfilename member that specifies the desired inputfile name.

    | **Outputs:**
    |    *inputs.outdir/metrics.txt:* output metrics for specified test image given by inputs.imagepath
    |    *inputs.loaddir/<inputs.imagepath>.jpg:* test image with detected bounding boxes, classes and confidence scores
    |    *inputs.loaddir/<inputs.imagepath>.tif.txt:*  text file with bounding boxes, classes and confidence scores for all detections
    """

    # Read input file options
    parser  = argparse.ArgumentParser(description='Input filename');
    parser.add_argument('inputfilename',\
                        metavar='inputfilename',type=str,\
                        help='Filename of the input file')
    args   = parser.parse_args()
    opt    = InputFile(args);
    #os.system('rm -rf ' + opt.outdir)
    os.makedirs(opt.outdir, exist_ok=True)
    opt.printInputs();
    
    # Setup testing problem
    model      = Darknet(opt)
    dataloader = ImageFolder(opt)
    tester     = NetworkTester(model,dataloader,opt)
    
    # Object detection on test dataset
    tester.detect()
    tester.plotDetection()

    # Metrics
    if opt.plot_flag:
        from scoring import score
        score.score(opt.outdir, opt.targetspath, opt.outdir)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    detect()
    torch.cuda.empty_cache()
