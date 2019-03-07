import argparse
import torch
from yolov3.utils.utils import assert_single_gpu_support
from yolov3.src.InputFile import *
from yolov3.src.models import *
from yolov3.src.datasets.ImageFolder import *
from yolov3.src.NetworkTester import *
from yolov3.src.scoring.score import *

def detect():
    """
    Main driver script for testing the YOLOv3 network.

    **Inputs**

    ----------
    args : command line arguments
        command line arguments used in shell call for this main driver script. args must have a inputfilename member that specifies the desired inputfile name.

    **Outputs**

    ----------
    inputs.outdir/metrics.txt : text file
        output metrics for specified test image given by inputs.imagepath
    inputs.loaddir/<inputs.imagepath>.jpg : jpeg image
        test image with detected bounding boxes, classes and confidence scores
    inputs.loaddir/<inputs.imagepath>.tif.txt : text file
        text file with bounding boxes, classes and confidence scores for all detections
    """

    # Read input file options
    parser  = argparse.ArgumentParser(description='Input filename');
    parser.add_argument('inputfilename',\
                        metavar='inputfilename',type=str,\
                        help='Filename of the input file')
    args          = parser.parse_args()
    inputfilename = args.inputfilename
    opt           = InputFile(inputfilename);
    os.makedirs(opt.outdir, exist_ok=True)
    opt.printInputs();
    
    # Setup testing problem
    assert_single_gpu_support()
    model      = Darknet(opt.networkcfg, opt.imgsize)
    dataloader = ImageFolder(opt)
    tester     = NetworkTester(model,dataloader,opt)
    
    # Object detection on test dataset
    tester.detect()
    tester.plotDetection()

    # Metrics
    if opt.plot_flag:
        score(opt)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    detect()
    torch.cuda.empty_cache()
