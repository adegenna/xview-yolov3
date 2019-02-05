import csv
import json
import os
import time
import scipy.io
import numpy as np

from scoring.matching import Matching
from scoring.rectangle import Rectangle
from utils.datasetProcessing import *

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


class ScoringData():
    """
    Structure to package various intermediate data/calculations together for scoring purposes.
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
        self.average_precision_per_class = None
        self.per_class_p                 = None
        self.per_class_r                 = None
        self.per_class_rcount            = None
        self.per_file_class_data         = None
        self.num_gt_per_cls              = None
        self.attempted                   = None
        self.num_preds                   = None
        
    def parse_predictions(self):
        per_file_class_data = {}
        for i in self.gt_unique:
            per_file_class_data[i] = [[], []]
        num_gt_per_cls = np.zeros((self.max_gt_cls))
        attempted      = np.zeros(self.max_gt_cls)
        for file_ind in range(len(self.pchips)):
            print(self.pchips[file_ind])
            det_box    = self.boxes_dict[self.pchips[file_ind]][:, :4]
            det_scores = self.boxes_dict[self.pchips[file_ind]][:, 5]
            det_cls    = self.boxes_dict[self.pchips[file_ind]][:, 4]
            gt_box     = self.gt_coords[(self.gt_chips == self.pchips[file_ind]).flatten()]
            gt_cls     = self.gt_classes[(self.gt_chips == self.pchips[file_ind])]
            for i in self.gt_unique:
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
                rects_matched, gt_matched  = matching.greedy_match(self.iou_threshold)
                # we aggregate confidence scores, rectangles, and num_gt across classes
                # per_file_class_data[i][0] += det_scores[det_cls == i].tolist()
                per_file_class_data[i][1] += rects_matched
                num_gt_per_cls[i]         += len(gt_matched)
        # Set three main results
        self.per_file_class_data               = per_file_class_data
        self.num_gt_per_cls                    = num_gt_per_cls
        self.attempted                         = attempted

    def calculate_per_class_metrics(self):
        average_precision_per_class = np.ones(self.max_gt_cls) * float('nan')
        per_class_p                 = np.ones(self.max_gt_cls) * float('nan')
        per_class_r                 = np.ones(self.max_gt_cls) * float('nan')
        per_class_rcount            = np.ones(self.max_gt_cls) * float('nan')
        for i in self.gt_unique:
            scores        = np.array(self.per_file_class_data[i][0])
            rects_matched = np.array(self.per_file_class_data[i][1])
            if self.num_gt_per_cls[i] != 0:
                sorted_indices = np.argsort(scores)[::-1]
                tp_sum         = np.cumsum(rects_matched[sorted_indices])
                fp_sum         = np.cumsum(np.logical_not(rects_matched[sorted_indices]))
                precision      = tp_sum / (tp_sum + fp_sum + np.spacing(1))
                recall         = tp_sum / self.num_gt_per_cls[i]
                per_class_p[i] = np.sum(rects_matched) / len(rects_matched)
                per_class_r[i] = np.sum(rects_matched) / self.num_gt_per_cls[i]
                per_class_rcount[i] = np.sum(rects_matched)
                ap                  = ap_from_pr(precision, recall)
            else:
                ap = 0
            average_precision_per_class[i] = ap
        # Set main results
        self.average_precision_per_class = average_precision_per_class
        self.per_class_p                 = per_class_p
        self.per_class_r                 = per_class_r
        self.per_class_rcount            = per_class_rcount

    def extract_detections_from_file(self,path_predictions):
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
        # Set main results
        self.boxes_dict = boxes_dict
        self.pchips     = pchips
        self.num_preds  = num_preds

    def calculate_dataset_splits(self,opt):
        common_classes, rare_classes                 = determine_common_and_rare_classes(opt)
        small_classes, medium_classes, large_classes = determine_small_medium_large_classes(opt)
        # Set main results
        self.splits = {
            'map/small': small_classes,
            'map/medium': medium_classes,
            'map/large': large_classes,
            'map/common': common_classes,
            'map/rare': rare_classes
        }

    def compute_final_statistical_metrics(self):
        vals = {}
        vals['map']       = np.nanmean(self.average_precision_per_class)
        vals['map_score'] = np.nanmean(self.per_class_p)
        vals['mar_score'] = np.nanmean(self.per_class_r)
        a = np.concatenate(
            (self.average_precision_per_class, self.per_class_p, self.per_class_r, self.per_class_rcount, self.num_gt_per_cls)).reshape(5, self.max_gt_cls)
        for i in self.splits.keys():
            vals[i] = np.nanmean(self.average_precision_per_class[self.splits[i]])
        v2 = np.zeros((len(self.gt_unique), 5))
        for i, j in enumerate(self.gt_unique):
            v2[i, 0] = j
            v2[i, 1] = self.attempted[j]
            v2[i, 2] = self.per_class_rcount[j]
            v2[i, 3] = self.num_gt_per_cls[j]
            v2[i, 4] = self.average_precision_per_class[j]
        for i in self.gt_unique:
            vals[int(i)] = self.average_precision_per_class[int(i)]
        vals['f1'] = 2 / ((1 / (np.spacing(1) + vals['map_score']))
                          + (1 / (np.spacing(1) + vals['mar_score'])))
        return vals,v2

