User Instructions
=====================

The purpose of this section is to provide detailed, step-by-step
instructions on how to use the important interfaces for our
codebase.

Importing YOLOv3
---------------------

#. Make sure you activate the conda environment was described in the
   Installation section::

     >>> conda activate [envname]

#. In your Python driver script, make sure that the yolov3 filepath is
   in the Python search path. For example, if the full filepath to the
   yolov3 repo is ``/full/path/to/topdir/yolov3/``, then you would do
   this as::

     import sys
     sys.path.append('/full/path/to/topdir/')

#. The yolov3 project uses the Python ``import`` system to load all
   source code in ``yolov3/src/`` and ``yolov3/utils/`` in a
   heirarchical manner. You may therefore import this codebase as a
   module; for example::

     import yolov3
     darknet = yolov3.src.models.Darknet(<network_cfg_file> , <imgsize>)

Unit Tests
---------------------

Unit tests are maintained in the ``yolov3/tests`` subdirectory. To run
these, navigate to the directory that contains yolov3, and run the
unittests as a Python package from the shell::

  >>> python3 -m yolov3.tests.unittests


Training Example
---------------------

An example driver script demonstrating proper interface usage for
network training is provided in ``scripts/train.py``, together with
the ``scripts/input_train.dat`` input file.

.. note:: To run this example, certain options must be set in the inputfile.
   In particular, please ensure that the following pieces of data exist
   and are specified in the input file:
     (1) ``targetspath`` : filepath to the target metadata file
     (2) ``traindir`` : filepath to the training image dataset
     (3) ``networkcfg`` : YOLOv3 network configuration file
     (4) ``class_path`` : filepath to the class names/labels file
   Please consult the Code Documentation on the InputFile class for further details on these and other necessary options.

To run this example, do the
following:

#. Edit the ``scripts/input_train.dat`` input file so that those
   settings involving filepaths accurately reflect the data/filepaths
   on your machine.

#. Navigate to the directory that contains yolov3, and issue the
   following shell command to run the script as a package::

  >>> python3 -m yolov3.scripts.train yolov3/scripts/input_train.dat
  
  .. note:: Mulitple-GPU support is currently not available for any
   part of this software. Please run in either CPU or single-GPU mode
   only. If you have multiple GPUs on your machine, you may use the
   CUDA_VISIBLE_DEVICES flag to enforce single-GPU mode, e.g.::

     >>> CUDA_VISIBLE_DEVICES=0 python3 -m yolov3.scripts.train yolov3/scripts/input_train.dat

Detection Example
---------------------

An example driver script demonstrating proper interface usage for
network detection/testing is provided in ``scripts/detect.py``,
together with the ``scripts/input_detect.dat`` input file.

.. note:: To run this example, certain options must be set in the inputfile. For image detection only,
   set ``plot_flag = False`` and ensure that the following pieces of data exist
   and are specified in the input file:
     (1) ``imagepath`` : filepath to image dataset
     (2) ``networkcfg`` : YOLOv3 network configuration file
     (3) ``networksavefile`` : PyTorch saved neural network file (.pt)
   To additionally run with scoring, set ``plot_flag = True`` and ensure that these
   necessary options are specified in the input file as well:
     (1) ``targetspath`` : filepath to the target metadata file
     (2) ``class_path`` : filepath to the class names/labels file
     (3) ``class_mean`` : filepath to training data class mean statistics
     (4) ``class_sigma`` : filepath to training data class sigma statistics
   For more details, consult the Code Documentation on the InputFile class.

To run this example,
do the following:

#. Edit the ``scripts/input_detect.dat`` input file so that those
   settings involving filepaths accurately reflect the data/filepaths
   on your machine.

#. Navigate to the directory that contains yolov3, and issue the
   following shell command to run the script as a package::

  >>> python3 -m yolov3.scripts.detect yolov3/scripts/input_detect.dat
  
  .. note:: Mulitple-GPU support is currently not available for any
   part of this software. Please run in either CPU or single-GPU mode
   only. If you have multiple GPUs on your machine, you may use the
   CUDA_VISIBLE_DEVICES flag to enforce single-GPU mode, e.g.::

     >>> CUDA_VISIBLE_DEVICES=0 python3 -m yolov3.scripts.detect yolov3/scripts/input_detect.dat

