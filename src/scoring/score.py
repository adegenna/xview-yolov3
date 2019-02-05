# Copyright 2018 Defense Innovation Unit Experimental
# All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import json
import os
import time
import scipy.io
import numpy as np

from scoring.scoringfunctions import *
from utils.datasetProcessing import *

# @profile
def score(opt, iou_threshold=.5):
    """
      Compute metrics on a number of prediction files, given a folder of prediction files
      and a ground truth.  Primary metric is mean average precision (mAP).

      **Inputs**

      ----------
      opt : InputFile
            InputFile member specifying all user options. Note: prediction files in opt.outdir should have filename format 'XYZ.tif.txt', where 'XYZ.tif' is the xView TIFF file being predicted on. Prediction files should be in space-delimited csv format, with each line like (xmin ymin xmax ymax class_prediction score_prediction).

      iou_threshold : float 
            iou threshold (between 0 and 1) indicating the percentage iou required to count a prediction as a true positive

      **Outputs**

      -----------
      opt.outdir/metrics.txt : text file
          contains the scoring metrics in per-line format (metric/class_num: score_float)

      **Raises**
    
      ----------
      ValueError 
          Raised if there are files in the prediction folder that are not in the ground truth geojson. EG a prediction file is titled '15.tif.txt', but the file '15.tif' is not in the ground truth.

    """
    path_predictions = opt.outdir
    path_groundtruth = opt.targetspath
    path_output      = opt.outdir
    assert (iou_threshold < 1 and iou_threshold > 0)
    print('Computing mAP and associated metrics on test data...')
    ttime = time.time()

    scoring_data                    = ScoringData()
    scoring_data.extract_detections_from_file(path_predictions)
    gt_coords, gt_chips, gt_classes = get_labels_geojson(path_groundtruth)
    
    scoring_data.gt_coords  = gt_coords
    scoring_data.gt_chips   = gt_chips
    scoring_data.gt_classes = gt_classes
    scoring_data.gt_unique  = np.unique(gt_classes.astype(np.int64))
    scoring_data.max_gt_cls = 100
    scoring_data.iou_threshold = iou_threshold

    if set(scoring_data.pchips).issubset(set(scoring_data.gt_unique)):
        raise ValueError('The prediction files {%s} are not in the ground truth.' % str(set(scoring_data.pchips) - (set(scoring_data.gt_unique))))
    print("Number of Predictions: %d" % scoring_data.num_preds)
    print("Number of GT: %d" % np.sum(scoring_data.gt_classes.shape))

    scoring_data.parse_predictions()
    scoring_data.calculate_per_class_metrics()
    scoring_data.calculate_dataset_splits(opt)
        
    _, _, classes   = get_labels_geojson(opt.targetspath)
    n               = np.setdiff1d( np.unique(classes) , opt.invalid_class_list )
    num_classes     = len(n)
    with open(opt.class_path) as f:
        lines = f.readlines()
    map_dict = {}
    for i in range(num_classes):
        map_dict[lines[i].replace('\n','')] = scoring_data.average_precision_per_class[int(n[i])]
    print(np.nansum(scoring_data.per_class_rcount), map_dict)
    vals,v2 = scoring_data.compute_final_statistical_metrics()

    # Output results
    print("mAP: %f | mAP score: %f | mAR: %f | F1: %f" %
          (vals['map'], vals['map_score'], vals['mar_score'], vals['f1']))
    with open(path_output + '/metrics.txt', 'w') as f:
        for key in vals.keys():
           f.write("%s %f\n" % (str(key), vals[key]))
        # for key in vals.keys():
        #     f.write("%f\n" % (vals[key]))
        for i in range(len(v2)):
            f.write(('%g, ' * 5 + '\n') % (v2[i, 0], v2[i, 1], v2[i, 2], v2[i, 3], v2[i, 4]))
    print("Final time: %s" % str(time.time() - ttime))


    
