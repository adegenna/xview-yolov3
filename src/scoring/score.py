"""
Copyright 2018 Defense Innovation Unit Experimental
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import csv
import json
import os
import time
import scipy.io
import numpy as np
from tqdm import tqdm

from scoring.matching import Matching
from scoring.rectangle import Rectangle
from utils.datasetProcessing import *

"""
  Scoring code to calculate per-class precision and mean average precision.

  Args:
      predictions: a folder path of prediction files.
        Prediction files should have filename format 'XYZ.tif.txt',
        where 'XYZ.tif' is the xView TIFF file being predicted on.
        Prediction files should be in space-delimited csv format, with each
        line like (xmin ymin xmax ymax class_prediction score_prediction).
        ie ("predictions/")

      groundtruth: a filepath to ground truth labels (GeoJSON format)
        ie ("ground_truth.geojson")

      output (-o): a folder path where output metrics are saved
        ie ("scores/")

  Outputs:
    Writes two files to the 'output' parameter folder: 'score.txt' and 'metrics.txt'
    'score.txt' contains a single floating point value output: mAP
    'metrics.txt' contains the remaining metrics in per-line format (metric/class_num: score_float)

"""

#@profile
def get_labels(fname):
    """
      Processes a WorldView3 GEOJSON file

      Args:
          fname: filepath to the GeoJson file.

      Outputs:
        Bounding box coordinate array, Chip-name array, and Classes array

    """
    with open(fname) as f:
        data = json.load(f)

    coords = np.zeros((len(data['features']), 4))
    chips = np.zeros((len(data['features'])), dtype="object")
    classes = np.zeros((len(data['features'])))

    for i in tqdm(range(len(data['features']))):
        if data['features'][i]['properties']['bounds_imcoords'] != []:
            b_id = data['features'][i]['properties']['image_id']
            val = np.array([int(num) for num in data['features'][i]['properties']['bounds_imcoords'].split(",")])
            chips[i] = b_id
            classes[i] = data['features'][i]['properties']['type_id']
            if val.shape[0] != 4:
                raise ValueError('A bounding box should have 4 entries!')
            else:
                coords[i] = val
        else:
            chips[i] = 'None'

    return coords, chips, classes


def convert_to_rectangle_list(coordinates):
    """
      Converts a list of coordinates to a list of rectangles

      Args:
          coordinates: a flattened list of bounding box coordinates in format
            (xmin,ymin,xmax,ymax)

      Outputs:
        A list of rectangles

    """
    rectangle_list = []
    number_of_rects = int(len(coordinates) / 4)
    for i in range(number_of_rects):
        rectangle_list.append(Rectangle(
            coordinates[4 * i], coordinates[4 * i + 1], coordinates[4 * i + 2],
            coordinates[4 * i + 3]))
    return rectangle_list


def ap_from_pr(p, r):
    """
      Calculates AP from precision and recall values as specified in
      the PASCAL VOC devkit.

      Args:
          p: an array of precision values
          r: an array of recall values

      Outputs:
        An average precision value

    """
    r = np.concatenate([[0], r, [1]])
    p = np.concatenate([[0], p, [0]])
    for i in range(p.shape[0] - 2, 0, -1):
        if p[i] > p[i - 1]:
            p[i - 1] = p[i]

    i = np.where(r[1:] != r[:len(r) - 1])[0] + 1
    ap = np.sum(
        (r[i] - r[i - 1]) * p[i])

    return ap


def extract_detections_from_file(path_predictions):
    boxes_dict = {}
    pchips     = []
    num_preds  = 0
    for file in os.listdir(path_predictions):
        fname = file.split(".txt")[0]
        if ( file.endswith(".tif.txt") ):
            pchips.append(fname)
            with open(path_predictions + file, 'r') as f:
                arr = np.array(list(csv.reader(f, delimiter=" ")))
                if arr.shape[0] == 0:
                    # If the file is empty, we fill it in with an array of zeros
                    boxes_dict[fname] = np.array([[0, 0, 0, 0, 0, 0]])
                    num_preds += 1
                else:
                    arr = arr[:, :6].astype(np.float64)
                    threshold = 0
                    arr = arr[arr[:, 5] > threshold]
                    num_preds += arr.shape[0]
                    if np.any(arr[:, :4] < 0):
                        raise ValueError('Bounding boxes cannot be negative.')
                    if np.any(arr[:, 5] < 0) or np.any(arr[:, 5] > 1):
                        raise ValueError('Confidence scores should be between 0 and 1.')
                    boxes_dict[fname] = arr[:, :6]
    pchips = sorted(pchips)
    return boxes_dict, pchips, num_preds

class Scoring_Data():
    """
    Structure to package various intermediate data together for scoring purposes.
    """
    def __init__(self):
        self.boxes_dict    = None
        self.pchips        = None
        self.gt_unique     = None
        self.max_gt_cls    = None
        self.gt_coords     = None
        self.gt_classes    = None
        self.gt_chips      = None
        self.iou_threshold = None

        
def parse_predictions(scoring_data):
    per_file_class_data = {}
    for i in scoring_data.gt_unique:
        per_file_class_data[i] = [[], []]
    num_gt_per_cls = np.zeros((scoring_data.max_gt_cls))
    attempted      = np.zeros(scoring_data.max_gt_cls)
    for file_ind in range(len(scoring_data.pchips)):
        print(scoring_data.pchips[file_ind])
        det_box    = scoring_data.boxes_dict[scoring_data.pchips[file_ind]][:, :4]
        det_scores = scoring_data.boxes_dict[scoring_data.pchips[file_ind]][:, 5]
        det_cls    = scoring_data.boxes_dict[scoring_data.pchips[file_ind]][:, 4]
        gt_box     = scoring_data.gt_coords[(scoring_data.gt_chips == scoring_data.pchips[file_ind]).flatten()]
        gt_cls     = scoring_data.gt_classes[(scoring_data.gt_chips == scoring_data.pchips[file_ind])]
        for i in scoring_data.gt_unique:
            s                          = det_scores[det_cls == i]
            ssort                      = np.argsort(s)[::-1]
            per_file_class_data[i][0] += s[ssort].tolist()
            gt_box_i_cls               = gt_box[gt_cls == i].flatten().tolist()
            det_box_i_cls              = det_box[det_cls == i]
            det_box_i_cls              = det_box_i_cls[ssort].flatten().tolist()
            gt_rects                   = convert_to_rectangle_list(gt_box_i_cls)
            rects                      = convert_to_rectangle_list(det_box_i_cls)
            attempted[i]              += len(rects)
            matching                   = Matching(gt_rects, rects)
            rects_matched, gt_matched  = matching.greedy_match(scoring_data.iou_threshold)
            # we aggregate confidence scores, rectangles, and num_gt across classes
            # per_file_class_data[i][0] += det_scores[det_cls == i].tolist()
            per_file_class_data[i][1] += rects_matched
            num_gt_per_cls[i]         += len(gt_matched)
    return per_file_class_data, num_gt_per_cls, attempted


# @profile
def score(opt, iou_threshold=.5):
    """
      Compute metrics on a number of prediction files, given a folder of prediction files
      and a ground truth.  Primary metric is mean average precision (mAP).

      **Inputs**

      ----------
      opt : InputFile
            InputFile member specifying all user options

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
    boxes_dict, pchips, num_preds   = extract_detections_from_file(path_predictions)
    gt_coords, gt_chips, gt_classes = get_labels(path_groundtruth)
    gt_unique  = np.unique(gt_classes.astype(np.int64))
    max_gt_cls = 100

    if set(pchips).issubset(set(gt_unique)):
        raise ValueError('The prediction files {%s} are not in the ground truth.' % str(set(pchips) - (set(gt_unique))))
    print("Number of Predictions: %d" % num_preds)
    print("Number of GT: %d" % np.sum(gt_classes.shape))

    scoring_data               = Scoring_Data()
    scoring_data.boxes_dict    = boxes_dict
    scoring_data.pchips        = pchips
    scoring_data.gt_unique     = gt_unique
    scoring_data.max_gt_cls    = max_gt_cls
    scoring_data.gt_coords     = gt_coords
    scoring_data.gt_classes    = gt_classes
    scoring_data.gt_chips      = gt_chips
    scoring_data.iou_threshold = iou_threshold

    per_file_class_data, num_gt_per_cls, attempted = parse_predictions(scoring_data)

    average_precision_per_class = np.ones(max_gt_cls) * float('nan')
    per_class_p                 = np.ones(max_gt_cls) * float('nan')
    per_class_r                 = np.ones(max_gt_cls) * float('nan')
    per_class_rcount            = np.ones(max_gt_cls) * float('nan')

    for i in gt_unique:
        scores = np.array(per_file_class_data[i][0])
        rects_matched = np.array(per_file_class_data[i][1])

        if num_gt_per_cls[i] != 0:
            sorted_indices = np.argsort(scores)[::-1]
            tp_sum = np.cumsum(rects_matched[sorted_indices])
            fp_sum = np.cumsum(np.logical_not(rects_matched[sorted_indices]))
            precision = tp_sum / (tp_sum + fp_sum + np.spacing(1))
            recall = tp_sum / num_gt_per_cls[i]
            per_class_p[i] = np.sum(rects_matched) / len(rects_matched)
            per_class_r[i] = np.sum(rects_matched) / num_gt_per_cls[i]
            per_class_rcount[i] = np.sum(rects_matched)
            ap = ap_from_pr(precision, recall)
        else:
            ap = 0
        average_precision_per_class[i] = ap

    # metric splits
    common_classes, rare_classes                 = determine_common_and_rare_classes(opt)
    small_classes, medium_classes, large_classes = determine_small_medium_large_classes(opt)
    
    metric_keys = ['map', 'map/small', 'map/medium', 'map/large',
                   'map/common', 'map/rare']

    splits = {
        'map/small': small_classes,
        'map/medium': medium_classes,
        'map/large': large_classes,

        'map/common': common_classes,
        'map/rare': rare_classes
    }
    
    _, _, classes   = get_labels_geojson(opt.targetspath)
    n               = np.setdiff1d( np.unique(classes) , opt.invalid_class_list )
    num_classes     = len(n)
    with open(opt.class_path) as f:
        lines = f.readlines()

    map_dict = {}
    for i in range(num_classes):
        map_dict[lines[i].replace('\n','')] = average_precision_per_class[int(n[i])]

    print(np.nansum(per_class_rcount), map_dict)
    vals = {}
    vals['map'] = np.nanmean(average_precision_per_class)
    vals['map_score'] = np.nanmean(per_class_p)
    vals['mar_score'] = np.nanmean(per_class_r)

    a = np.concatenate(
        (average_precision_per_class, per_class_p, per_class_r, per_class_rcount, num_gt_per_cls)).reshape(5, max_gt_cls)

    for i in splits.keys():
        vals[i] = np.nanmean(average_precision_per_class[splits[i]])
    
    v2 = np.zeros((len(gt_unique), 5))
    for i, j in enumerate(gt_unique):
        v2[i, 0] = j
        v2[i, 1] = attempted[j]
        v2[i, 2] = per_class_rcount[j]
        v2[i, 3] = num_gt_per_cls[j]
        v2[i, 4] = average_precision_per_class[j]

    for i in gt_unique:
        vals[int(i)] = average_precision_per_class[int(i)]

    vals['f1'] = 2 / ((1 / (np.spacing(1) + vals['map_score']))
                      + (1 / (np.spacing(1) + vals['mar_score'])))

    print("mAP: %f | mAP score: %f | mAR: %f | F1: %f" %
          (vals['map'], vals['map_score'], vals['mar_score'], vals['f1']))

    # with open(path_output + '/score.txt', 'w') as f:
    #     f.write(str("%.8f" % vals['map']))
    #
    with open(path_output + '/metrics.txt', 'w') as f:
        for key in vals.keys():
           f.write("%s %f\n" % (str(key), vals[key]))
        # for key in vals.keys():
        #     f.write("%f\n" % (vals[key]))
        for i in range(len(v2)):
            f.write(('%g, ' * 5 + '\n') % (v2[i, 0], v2[i, 1], v2[i, 2], v2[i, 3], v2[i, 4]))

    print("Final time: %s" % str(time.time() - ttime))


    
