Code Docs
=====================

Src/
---------------------
.. automodule:: src.train
   :members:
      
.. automodule:: src.detect
   :members:

.. automodule:: src.InputFile
   :members:

.. automodule:: src.models
   :members:

.. automodule:: src.NetworkTrainer
   :members:

.. automodule:: src.NetworkTester
   :members:

.. automodule:: src.targets
   :members: Target, fcn_sigma_rejection, per_class_stats

.. automodule:: src.datasets
   :members: ListDataset, ImageFolder, pickRandomPoints, augmentHSV, resize_square, random_affine

.. automodule:: src.scoring
   :members: get_labels, convert_to_rectangle_list, ap_from_pr, score, safe_divide, compute_statistics_given_rectangle_matches, compute_precision_recall_given_image_statistics_list, compute_average_precision_recall_given_precision_recall_dict, convert_to_rectangle_list, compute_average_precision_recall, Matching, cartesian, Rectangle
	     
Utils/
---------------------
.. automodule:: utils.datasetProcessing
   :members:

.. automodule:: utils.utils
   :members:

.. automodule:: utils.utils_xview
   :members:

Tests/
---------------------
.. automodule:: tests.unittests
   :members:
