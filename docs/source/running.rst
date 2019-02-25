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
your inputfile is also in that directory, and simply run::

  python train.py train.py <input_file_name>

Notes on the training inputfile:

#. ``datadir`` : This option sets the full filepath to the location on
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

#. ``invalidclasslist`` : This option was added to give the user the
   ability to specify a list of classes referred to in the
   ``targetspath`` metadata file that either are not present or need
   to be excluded from the training dataset. For example, the xView
   dataset makes reference to classes 75 and 82 in the .geojson target
   file, but these are ``None`` classes.

   If there are no "invalid" classes in your target metadata file,
   then simply leave this option blank.

Here is an example inputfile for training that demonstrates correct
option specification. In this case, the user is asking the software to
precompute the YOLO architecture::

  [TRAIN]
  loaddir          = /full/path/to/xview-yolov3/checkpoints/
  outdir           = /full/path/to/xview-yolov3/output/
  targetspath      = /full/path/to/targetsdir/xView_train.geojson
  targetfiletype   = json
  traindir         = /full/path/to/traindir/
  epochs           = 300
  epochstart       = 0
  batchsize        = 8
  networkcfg       = /full/path/to/xview-yolov3/cfg/yolov3_template.cfg
  imgsize          = 800
  resume           = False
  invalid_class_list         = 75,82
  boundingboxclusters        = 30
  computeboundingboxclusters = False

Testing
---------------------

Network testing is handled by the driver script ``src/detect.py``. To
run this, make sure you are in the ``src/`` subdirectory, and that
your inputfile is also in that directory, and simply run::

  python detect.py train.py <input_file_name>

Notes on the testing inputfile:

