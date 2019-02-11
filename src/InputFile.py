import numpy as np
from pprint import pprint
import sys

# Auxiliary class for packaging input file options
class InputFile():
    """
    Class for packaging all input/config file options together.

    **Inputs**

    ----------
    args : command line arguments 
        (passed to constructor at runtime) command line arguments used in shell call for main driver script. args must have a inputfilename member that specifies the desired inputfile name. 

    **Options**

    ----------
    inputtype : string
        Options are `train` or `detect`
    projdir : string
        Absolute path to project directory
    datadir : string
        Absolute path to data directory
    loaddir : string
        Absolute path to load directory
    outdir : string
        Absolute path to output directory
    targetspath : string
        Absolute path to target file
    targetfiletype : string
        Type of target file
    traindir : string
        Type of target file

    **Options (Train-Specific)**

    ----------
    traindir : string 
        Type of target file
    epochs : int
        Number of training epochs
    epochstart : int
        Starting epoch
    batchsize : int
        Training batch size
    networkcfg : string
        Network architecture file
    imgsize : int
        Base image crop size
    resume : bool
        Boolean value specifying whether training is resuming from previous iteration
    invalid_class_list : string (csv format)
        Comma-separated list of classes to be ignored from training data
    boundingboxclusters : int
        Desired number of bounding-box clusters for the YOLO architecture
    computeboundingboxclusters : bool
        Boolean value specifying whether to compute bounding box clusters

    **Options (Detect-Specific)**

    ----------
    imagepath : string
        Image path
    plotflag : bool
        Flag for plotting
    secondary_classifier : bool
        Boolean value specifying whether to use a secondary classifier
    networkcfg : string
        Network architecture file
    networksavefile : string
        Absolute path to trained YOLOv3 network file, saved by PyTorch (.pt)
    class_path : string
        Absolute path to class
    conf_thres : float
        Confidence threshold for detection
    cls_thres : float
        Class threshold for detection
    nms_thres : float
        NMS threshold
    batch_size : int
        Desired batchsize
    img_size : int
        Desired cropped image size
    rgb_mean : string
        Absolute path to dataset RGB mean file
    rgb_std : string
        Absolute path to dataset RGB standard deviation file
    class_mean : string
        Absolute path to class mean file
    class_sigma : string
        Absolute path to class standard deviation file
    invalid_class_list : string (csv format)
        Comma-separated list of classes to be ignored from training data
    """
    
    def __init__(self,args=[]):
        try:
            inputfilename        = args.inputfilename
            inputfilestream      = open(inputfilename)
            inputtype            = inputfilestream.readline().strip().split('= ')[1];
            self.projdir         = inputfilestream.readline().strip().split('= ')[1];
            self.datadir         = inputfilestream.readline().strip().split('= ')[1];
            self.loaddir         = inputfilestream.readline().strip().split('= ')[1];
            self.outdir          = inputfilestream.readline().strip().split('= ')[1];
            self.targetspath     = inputfilestream.readline().strip().split('= ')[1];
            self.targetfiletype  = inputfilestream.readline().strip().split('= ')[1];
            if (inputtype == "train"):
                self.readTrainingInputfile(inputfilestream);
            elif (inputtype == "detect"):
                self.readDetectInputfile(inputfilestream);
            inputfilestream.close();
        except:
            print("Using no input file (blank initialization).")
    def printInputs(self):
        """
        Method to print all config options.
        """
        attrs = vars(self);
        print('\n');
        print("********************* INPUTS *********************")
        print('\n'.join("%s: %s" % item for item in attrs.items()))
        print("**************************************************")
        print('\n');
    def readTrainingInputfile(self,inputfilestream):
        """
        Method to read config options from a training inputfile.

        | **Inputs:**
        |    *inputfilestream:* specified inputfilestream.
        """
        self.traindir        = inputfilestream.readline().strip().split('= ')[1];
        self.epochs          = int(inputfilestream.readline().strip().split('= ')[1]);
        self.epochstart      = int(inputfilestream.readline().strip().split('= ')[1]);
        self.batchsize       = int(inputfilestream.readline().strip().split('= ')[1]);
        self.networkcfg      = inputfilestream.readline().strip().split('= ')[1];
        self.imgsize         = int(inputfilestream.readline().strip().split('= ')[1]);
        resume               = inputfilestream.readline().strip().split('= ')[1];
        self.resume          = ((resume == "True") | (resume == "true"));
        invalid_class_list   = inputfilestream.readline().strip().split('= ')[1]
        self.invalid_class_list         = np.array( invalid_class_list.split(',') , dtype='int' )
        self.boundingboxclusters        = int(inputfilestream.readline().strip().split('= ')[1]);
        computeboundingboxclusters      = inputfilestream.readline().strip().split('= ')[1];
        self.computeboundingboxclusters = ((computeboundingboxclusters == "True") | (computeboundingboxclusters == "true"));
    def readDetectInputfile(self,inputfilestream):
        """
        Method to read config options from a detection inputfile

        | **Inputs:**
        |    *inputfilestream:* specified inputfilestream.
        """
        self.imagepath       = inputfilestream.readline().strip().split('= ')[1];
        plotflag             = inputfilestream.readline().strip().split('= ')[1];
        self.plot_flag       = ((plotflag == "True") | (plotflag == "true"));
        secondary_classifier       = inputfilestream.readline().strip().split('= ')[1];
        self.secondary_classifier  = ((secondary_classifier == "True") | (secondary_classifier == "true"));
        self.networkcfg      = inputfilestream.readline().strip().split('= ')[1];
        self.networksavefile = inputfilestream.readline().strip().split('= ')[1];
        self.class_path      = inputfilestream.readline().strip().split('= ')[1];
        self.conf_thres      = float(inputfilestream.readline().strip().split('= ')[1]);
        self.cls_thres       = float(inputfilestream.readline().strip().split('= ')[1]);
        self.nms_thres       = float(inputfilestream.readline().strip().split('= ')[1]);
        self.batch_size      = int(inputfilestream.readline().strip().split('= ')[1]);
        self.imgsize         = int(inputfilestream.readline().strip().split('= ')[1]);
        self.rgb_mean        = inputfilestream.readline().strip().split('= ')[1];
        self.rgb_std         = inputfilestream.readline().strip().split('= ')[1];
        self.class_mean      = inputfilestream.readline().strip().split('= ')[1];
        self.class_sigma     = inputfilestream.readline().strip().split('= ')[1];
        invalid_class_list   = inputfilestream.readline().strip().split('= ')[1]
        self.invalid_class_list  = np.array( invalid_class_list.split(',') , dtype='int' )
