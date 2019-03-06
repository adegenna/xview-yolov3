User Instructions
=====================

The purpose of this document is to provide detailed, step-by-step
instructions on how to run the code in our codebase related to (1)
training a network and (2) testing it.

Assuming you have set-up and installed all Python requirements
correctly, all that remains is to ensure that you have properly
formatted inputfiles. Consult the Code Docs section for a full
description of the options. Below, we give some advice on a few of the
components that are needed to avoid potential pitfalls.


Training
---------------------

Network training is handled by the driver script ``src/train.py``. To
run this, make sure you are in the ``src/`` subdirectory, and that
your inputfile is also in that directory.

CPU mode::

  python3 train.py <input_file_name>

Single
GPU mode::
  CUDA_VISIBLE_DEVICES=0 python3 train.py <input_file_name>
  
.. note:: 
   Mulitple-GPU support is currently not available for any part of this software. Please run in either CPU or single-GPU mode only. Also, depending on your machine, the CUDA_VISIBLE_DEVICES flag may be set to an option other than 0, if for example you wish to use a GPU other than the first one.

Notes on the training inputfile:

#. ``traindir`` : This option sets the full filepath to the location on
   your machine where your training dataset resides. There should be
   nothing in this directory except the training image
   files. Currently supported image files are ``.tif`` and ``.bmp``.

#. ``targetspath`` : This option sets the full filepath to the
   location on your machine where the target metadata file for your
   training datset resides. Currently supported target file formats are 
   are ``.geojson``.

#. ``networkcfg`` : This option sets the full filepath to the location
   on your machine where the YOLOv3 network configuration file
   resides.  You have two options here: you may provide a complete
   YOLOv3 architecture file, or provide a network template file and
   request that the software precompute the architecture for you,
   prior to training.  We strongly recommend the latter option. To do
   this, provide the full filepath to the generic template file
   ``cfg/yolov3_template.cfg``, set ``computeboundingboxclusters =
   True`` in the inputfile, and provide the desired number of bounding
   box clusters (i.e. YOLO anchors) in the ``boundingboxclusters``
   argument of the inputfile.

#. ``imgsize`` : This integer value specifies the pixel length of a
   single image chip. Because of the specifics of the YOLOv3
   architecture, the only constraint is that this number must be a
   multiple of 32.

#. ``invalid_class_list`` : This option was added to give the user the
   ability to specify a list of classes referred to in the
   ``targetspath`` metadata file that either are not present or need
   to be excluded from the training dataset. For example, the xView
   dataset makes reference to classes 75 and 82 in the .geojson target
   file, but these are ``None`` classes.

   If there are no "invalid" classes in your target metadata file,
   then simply leave this option blank.

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

Here is an example inputfile for training that demonstrates correct
option specification. In this case, the user is asking the software to
precompute the YOLO architecture::

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

Testing
---------------------

Network testing is handled by the driver script ``src/detect.py``. To
run this, make sure you are in the ``src/`` subdirectory, and that
your inputfile is also in that directory.

CPU mode::

  python3 detect.py <input_file_name>

Single
GPU mode::
  CUDA_VISIBLE_DEVICES=0 python3 detect.py <input_file_name>
  
.. note::
   Mulitple-GPU support is currently not available for any part of this software. Please run in either CPU or single-GPU mode only. Also, depending on your machine, the CUDA_VISIBLE_DEVICES flag may be set to an option other than 0, if for example you wish to use a GPU other than the first one.

Notes on the testing inputfile:

#. ``targetspath`` , ``invalid_class_list`` , ``imgsize`` ,
   ``class_path`` : Same notes apply as in the training case above.

#. ``imagepath`` : This option sets the full filepath to the location
   on your machine where your test dataset resides. There should be
   nothing in this directory except the test image files. Currently
   supported image files are ``.tif`` and ``.bmp``.

#. ``networkcfg`` : This option specifies the full filepath to a
   trained YOLOv3 configuration file. If you used the recommended
   input to this option in the training stage, then the code will have
   produced this file for you, saved as ``cfg/yolov3_custom.cfg``.

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
   
Here is an example inputfile for testing that demonstrates correct
option specification::

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
