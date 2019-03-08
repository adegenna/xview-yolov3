import numpy as np
import sys,os
import configparser

class InputFile():
    """Class for packaging all input/config file options together. 

    .. note:: 
         There are separate options required by InputFile depending on whether the intended goal is training or testing. The user must declare on the first line of the InputFile either ``[TRAIN]`` or ``[TEST]``, depending on their desired objective.

    **Inputs**

    ----------
    inputfilename : string 
        String specifying the desired inputfile name.     

    **Train Options**

    ----------
    Below are a list of options that must be specified in an inputfile
    of type ``[TRAIN]``:

    ==============================  ============  ===============================================
    option                          type          description
    ==============================  ============  ===============================================
    loaddir                         string        - Full path to load directory.
                                                  - Any pre-trained YOLOv3 PyTorch file (.pt) goes here 
    outdir                          string        - Full  path to output directory.
    targetspath                     string        - Full path to target file.
                                                  - Supported formats: ``.geojson``
                                                  - **[See notes below]**
    targetfiletype                  string        - Type of target file.
                                                  - Supported options: ``json``
    traindir                        string        - Full path to training image dataset
                                                  - Supported image types: ``.tif``, ``.bmp``
                                                  - **[See notes below]**
    epochs                          int           - Number of training epochs.
    epochstart                      int           - Starting epoch.
    batchsize                       int           - Training batch size.
    networkcfg                      string        - Full path to YOLOv3 network architecture file.
                                                  - Base template in ``yolov3/cfg/yolov3_template.cfg``
                                                  - **[See notes below]**
    imgsize                         int           - Base image chip size.
                                                  - Must be multiple of 32
    resume                          bool          - Specifies whether training is resuming from previous training.
    invalid_class_list              string        - .csv list of classes to be ignored from training data.
                                                  - **[See notes below]**
    boundingboxclusters             int           - Desired number of bounding-box clusters for the YOLO architecture.
    computeboundingboxclusters      bool          - Specifies whether to compute bounding box clusters.
    class_path                      string        - Full path to class names/labels file.
                                                  - **[See notes below]**
    sampling_weight                 string        - String specifying type of sampling weight. 
                                                  - Options are ``inverse_class_frequency`` and ``uniform``
                                                  - **[See notes below]**
    ==============================  ============  ===============================================

    Notes on ``[TRAIN]`` options above:

    #. ``traindir`` : If ``.tif`` images are used in the training data
       directory, then a .bmp copy is produced to be used in training.
    
    #. ``targetspath`` : Currently, the target metadata file must be
       formatted in .json format, similar to the xView .geojson
       format. Most important is that the target file must be
       compatible with the ``yolov3.utils.get_labels_geojson( )``
       function, which provides the implementation for retrieving
       object coordinates, corresponding filenames, and
       classes. Please consult the documentation for that function for
       further details.

    #. ``networkcfg`` : You have two options here: you may provide a
       complete YOLOv3 architecture file, or provide a network
       template file and request that the software precompute the
       architecture for you, prior to training.  We strongly recommend
       the latter option. To do this, provide the full filepath to the
       generic template file located in
       ``yolov3/cfg/yolov3_template.cfg``, set
       ``computeboundingboxclusters = True`` in the inputfile, and
       provide the desired number of bounding box clusters (i.e. YOLO
       anchors) in the ``boundingboxclusters`` argument of the
       inputfile. This will tell the software to precompute bounding
       box priors (anchors) for the training dataset, and a custom
       architecture file will be calculated and outputted to
       ``yolov3/cfg/yolov3_custom.cfg``, which may be used later in
       detection/testing.

    #. ``invalid_class_list`` : This option was added to give the user
       the ability to specify a list of classes referred to in the
       ``targetspath`` metadata file that either are not present or
       need to be excluded from the training dataset. For example, the
       xView dataset makes reference to classes 75 and 82 in the
       .geojson target file, but these are ``None`` classes. If there
       are no "invalid" classes in your target metadata file, then
       simply leave this option blank.

    #. ``class_path`` : This file is a comma-separated list of all classes
       and any associated numeric labels. For example, the xView dataset
       contains 60 classes, with associated labels ranging from 11
       to 94. Thus, the ``class_path`` file for the xView dataset would be
       a 60-line .csv file that would look as follows::

         Fixed-wing Aircraft , 11
         Small Aircraft , 12
         Cargo Plane , 13
         ...
         Tower , 94

    #. ``sampling_weight`` : This option sets how images are weighted for
       random selection at runtime during the training routine. Options
       are ``inverse_class_frequency`` and ``uniform``. The former weights
       an image by the sum of the inverse of the class frequencies of all
       its objects; the latter weights all images uniformly.
    
    **Test Options**

    ----------
    Below are a list of options that must be specified in an inputfile
    of type ``[TEST]``:

    ==============================  ============  ===============================================
    option                          data type     meaning
    ==============================  ============  ===============================================
    loaddir                         string        - Full path to load directory.
    outdir                          string        - Full path to output directory.
    targetspath                     string        - Full path to target file.
                                                  - Only needed for scoring the object detections.
    targetfiletype                  string        - Type of target file.
                                                  - Supported options: ``json``
    imagepath                       string        - Full path to directory containing images.
    plot_flag                       bool          - Flag to indicate whether to plot and output object detections.
                                                  - **[See notes below]**
    networkcfg                      string        - Full path to network architecture file.
                                                  - **[See notes below]**
    networksavefile                 string        - Full path to trained YOLOv3 network file.
                                                  - **[See notes below]**
    class_path                      string        - Full path to .csv file containing list of classes/labels.
    conf_thres                      float         - Confidence threshold for detection.
    cls_thres                       float         - Class threshold for detection.
    nms_thres                       float         - NMS threshold.
    batch_size                      int           - Batch size.
    imgsize                         int           - Desired chip size.
    rgb_mean                        string        - Full path to dataset RGB mean file.
                                                  - **[See notes below]**
    rgb_std                         string        - Full path to dataset RGB standard deviation file.
                                                  - **[See notes below]**
    class_mean                      string        - Full path to class mean file.
                                                  - **[See notes below]**
    class_sigma                     string        - Full path to class standard deviation file.
                                                  - **[See notes below]**
    invalid_class_list              string        - Comma-separated list of classes to be ignored from training data.
    ==============================  ============  ===============================================

    Notes on the testing inputfile:

    #. ``targetspath`` , ``invalid_class_list`` , ``imgsize`` ,
       ``class_path`` : Same notes apply as in the training case above.

    #. ``imagepath`` : This option sets the full filepath to the location
       on your machine where your test dataset resides. There should be
       nothing in this directory except the test image files. Currently
       supported image files are ``.tif`` and ``.bmp``.
    
    #. ``networkcfg`` : This option specifies the full filepath to a
       trained YOLOv3 configuration file. If you used the recommended
       input to this option in the training stage, then the code will
       have produced this file for you, saved as
       ``cfg/yolov3_custom.cfg``. Otherwise, you will have to fill in
       the YOLO anchors yourself directly into a copy of the template
       file.
    
    #. ``networksavefile`` : This option specifies the full filepath to
       the PyTorch savefile (.pt extension) that contains all weights for
       the trained network.
     
    #. ``rgb_mean`` , ``rgb_std`` : These files contain RGB statistics
       that were computed on the training dataset by the training
       routine. Each of them is simply a 3-line file, where each line
       contains a single numeric value that is the mean (or standard
       deviation) of the respective RGB channel. These values are used to
       normalize any data that is fed into the network.
    
    #. ``class_mean`` , ``class_std`` : These files contain class
       statistics that were computed on the training dataset by the
       training routine. Each of these files contains N-lines, where N is
       the number of classes, and each line contains a comma-separated
       list of 4 values, corresponding to the mean (or standard deviation)
       of the width, height, area, and aspect ratio (in that order) of the
       respective class objects. These statistics are used as prior
       information to reduce false positives in the object detection
       stage.

    #. ``plot_flag`` : This option specifies whether you would like to
       score the object detections that are calculated.

    **Examples**

    --------
    To use this class, 
    follow this interface::
        input_file_object = InputFile('/full/path/to/input_file.dat')

    For the ``[TRAIN]`` case, here is an example
    of what ``input_file.dat`` might contain::
        [TRAIN]
        loaddir          = /full/path/to/loaddir/
        outdir           = /full/path/to/outdir/
        targetspath      = /full/path/to/targetsdir/xView_train.geojson
        targetfiletype   = json
        traindir         = /full/path/to/traindir/
        epochs           = 300
        epochstart       = 0
        batchsize        = 8
        networkcfg       = /full/path/to/networkdir/yolov3_template.cfg
        imgsize          = 800
        resume           = False
        invalid_class_list         = 75,82
        boundingboxclusters        = 30
        computeboundingboxclusters = False
        class_path       = /full/path/to/xview_names_and_labels.csv
        sampling_weight  = inverse_class_frequency

    For the ``[TEST]`` case, here is an example
    of what ``input_file.dat`` might contain::
      [TEST]
      loaddir              = /full/path/to/loaddir/
      outdir               = /full/path/to/outdir/
      targetspath          = /full/path/to/targetdir/xView_train.geojson
      targetfiletype       = json
      imagepath            = /full/path/to/testdir/
      plot_flag            = True
      networkcfg           = /full/path/to/networksavedir/yolov3_custom.cfg
      networksavefile      = /full/path/to/networksavedir/best.pt
      class_path           = /full/path/to/classpathdir/xview_names_and_labels.csv
      conf_thres           = 0.99
      cls_thres            = 0.05
      nms_thres            = 0.4
      batch_size           = 1
      imgsize              = 1632
      rgb_mean             = /full/path/to/statdir/training_rgb_mean.out
      rgb_std              = /full/path/to/statdir/training_rgb_std.out
      class_mean           = /full/path/to/statdir/training_class_mean.out
      class_sigma          = /full/path/to/statdir/training_class_sigma.out
      invalid_class_list   = 75,82

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
