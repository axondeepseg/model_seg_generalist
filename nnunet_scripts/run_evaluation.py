"""
This script is used to run evaluation on a generalist model trained in the 
nnUNet framework. It will compute the metrics for every test image and display 
results for every dataset separately, for both axon and myelin.
Author: Armand Collin
"""

import argparse
import json
import warnings
import cv2
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from monai.metrics import DiceMetric, MeanIoU

def compute_metrics(pred, gt, metric):
    """
    Compute the given metric for a single image
    Args:
        pred: the prediction image
        gt: the ground truth image
        metric: the metric to compute
    Returns:
        the computed metric
    """
    value = metric(pred, gt)
    return value.item()

def extract_binary_masks(mask):
    '''
    This function will take as input an 8-bit image containing both the axon 
    class (value should be ~255) and the myelin class (value should be ~127).
    This function will also convert the numpy arrays read by opencv to Tensors.
    '''
    # axonmyelin masks should always have 3 unique values
    if len(np.unique(mask)) > 3:
        warnings.warn('WARNING: more than 3 unique values in the mask')
    myelin_mask = np.where(np.logical_and(mask > 100, mask < 200), 1, 0)
    myelin_mask = torch.from_numpy(myelin_mask).float()
    axon_mask = np.where(mask > 200, 1, 0)
    axon_mask = torch.from_numpy(axon_mask).float()
    return axon_mask, myelin_mask
    
def print_metric(value, gt, metric, label, reverted_mapping):
    # modify gt name to add _0000 suffix before file extension
    gt_name = gt.name.split('.')[0] + '_0000.' + gt.name.split('.')[1]
    original_fname = reverted_mapping[gt_name]
    metric_name = metric.__class__.__name__
    print(f'{metric_name} for {label} in {gt.name} (aka {original_fname}): {value}')

def main():
    parser = argparse.ArgumentParser(description='Run evaluation on a generalist model')
    parser.add_argument('-p', '--pred_path', type=str, help='Path to the predictions folder (axonmyelin preds)')
    parser.add_argument('-m', '--mapping_path', type=str, help='Path to the filename mapping JSON file')
    parser.add_argument('-g', '--gt_path', type=str, help='Path to the GT folder (axonmyelin masks)')
    parser.add_argument('-o', '--output_path', type=str, help='Path to save the evaluation results')
    args = parser.parse_args()

    pred_path = Path(args.pred_path)
    gt_path = Path(args.gt_path)
    mapping = json.load(open(args.mapping_path))
    reverted_mapping = {v: k for k, v in mapping['images_ts'].items()}
    # there might be more test imgs than GTs; evaluation on labelled data only
    gts = [f for f in gt_path.glob('*.png')]
    metrics = [DiceMetric(), MeanIoU()]

    # iterate over the ground truths
    for gt in gts:
        # get the corresponding prediction
        pred = pred_path / gt.name
        pred_im = cv2.imread(str(pred), cv2.IMREAD_GRAYSCALE)[None]
        pred_ax, pred_my = extract_binary_masks(pred_im)
        gt_im = cv2.imread(str(gt), cv2.IMREAD_GRAYSCALE)[None]
        gt_ax, gt_my = extract_binary_masks(gt_im)

        # compute the metrics
        for metric in metrics:
            # Compute metrics for axon
            value_ax = compute_metrics([pred_ax], [gt_ax], metric)
            print_metric(value_ax, gt, metric, 'axon', reverted_mapping)

            # Compute metrics for myelin
            value_my = compute_metrics([pred_my], [gt_my], metric)
            print_metric(value_my, gt, metric, 'myelin', reverted_mapping)

if __name__ == '__main__':
    main()

