import numpy as np
import sys,os
import configparser

class InputFile():
    """
    Class for packaging all input/config file options together.

    .. note:: 
         There are separate options required by InputFile depending on whether the intended goal is training or testing. The user must declare on the first line of the InputFile either ``[TRAIN]`` or ``[TEST]``, depending on their desired objective.

    **Inputs**

    ----------
    inputfilename : string 
        String specifying the desired inputfile name. 

    **Train Options**

    ----------
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
    class_path : string
        Absolute path to class

    **Test Options**

    ----------
    loaddir : string
        Absolute path to load directory
    outdir : string
        Absolute path to output directory
    targetspath : string
        Absolute path to target file
    targetfiletype : string
        Type of target file
    imagepath : string
        Image path
    plot_flag : bool
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
    imgsize : int
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

    def __init__(self, inputfile):
        config    = configparser.ConfigParser()
        config.read(inputfile)
        keys,vals = self.assert_everything_included(config)
        self.set_options(keys,vals)
        self.convert_strings_to_appropriate_datatypes()

    def assert_everything_included(self,config):
        # Check that correct section header is used
        self.inputtype = config.sections()[0]
        try:
            assert( (self.inputtype == 'TRAIN') | (self.inputtype == 'TEST') )
        except AssertionError as e:
            e.args += ('Please specify as the inputfile section header either [TRAIN] or [TEST]',)
            raise
        # Get keys
        keys = [k for k in config[self.inputtype]]
        vals = [config[self.inputtype][v] for v in keys]
        if (self.inputtype == 'TRAIN'):
            self.set_necessary_keys_train()
        elif (self.inputtype == 'TEST'):
            self.set_necessary_keys_test()
        # Check that all necessary keys are included
        for k in self.necessary_keys:
            try:
                assert( any(k in s for s in keys) )
            except AssertionError as e:
                e.args += ('The following necessary argument was not specified in the input file: ' + k,)
                raise
        return keys,vals

    def set_options(self,keys,vals):
        # Set all options
        for i,k in enumerate(keys):
            setattr(self,k,vals[i])

    def convert_strings_to_appropriate_datatypes(self):
        if (self.invalid_class_list):
            self.invalid_class_list  = np.array([int(x) for x in self.invalid_class_list.split(',')] , dtype='int')
        else:
            self.invalid_class_list = np.array([])
        # Train-specific
        if (self.inputtype == "TRAIN"):
            self.epochs     = int(self.epochs)
            self.epochstart = int(self.epochstart)
            self.batchsize  = int(self.batchsize)
            self.imgsize    = int(self.imgsize)
            self.resume     = ((self.resume == "True") | (self.resume == "true"))
            self.boundingboxclusters = int(self.boundingboxclusters)
            self.computeboundingboxclusters = ((self.computeboundingboxclusters == "True") | (self.computeboundingboxclusters == "true"))
        # Test-specific
        elif (self.inputtype == "TEST"):
            self.plot_flag = ((self.plot_flag == "True") | (self.plot_flag == "true"))
            self.secondary_classifier = ((self.secondary_classifier == "True") | (self.secondary_classifier == "true"))
            self.conf_thres = float(self.conf_thres)
            self.cls_thres  = float(self.cls_thres)
            self.nms_thres  = float(self.nms_thres)
            self.batch_size = int(self.batch_size)
            self.imgsize    = int(self.imgsize)

    def set_necessary_keys_train(self):
        # Set list defining necessary keys for training
        self.necessary_keys = ['loaddir', \
                               'outdir', \
                               'targetspath', \
                               'targetfiletype', \
                               'traindir', \
                               'epochs', \
                               'epochstart', \
                               'batchsize', \
                               'networkcfg', \
                               'imgsize', \
                               'resume', \
                               'invalid_class_list', \
                               'boundingboxclusters', \
                               'computeboundingboxclusters', \
                               'class_path', \
                               'sampling_weight']

    def set_necessary_keys_test(self):
        # Set list defining necessary keys for testing
        self.necessary_keys = ['loaddir', \
                               'outdir', \
                               'targetspath', \
                               'targetfiletype', \
                               'imagepath', \
                               'plot_flag', \
                               'secondary_classifier', \
                               'networkcfg', \
                               'networksavefile', \
                               'class_path', \
                               'conf_thres', \
                               'cls_thres', \
                               'nms_thres', \
                               'batch_size', \
                               'imgsize', \
                               'rgb_mean', \
                               'rgb_std', \
                               'class_mean', \
                               'class_sigma', \
                               'invalid_class_list']

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
