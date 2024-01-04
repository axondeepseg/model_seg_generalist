"""
This script is used to run evaluation on a generalist model trained in the 
nnUNet framework. It will compute the metrics for every test image and display 
results for every dataset separately.
Author: Armand Collin
"""

import argparse
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
    return value

def main():
    parser = argparse.ArgumentParser(description='Run evaluation on a generalist model')
    parser.add_argument('-p', '--pred_path', type=str, help='Path to the predictions folder')
    parser.add_argument('-m', '--mapping_path', type=str, help='Path to the filename mapping JSON file')
    parser.add_argument('-g', '--gt_path', type=str, help='Path to the GT folder')
    parser.add_argument('-o', '--output_path', type=str, help='Path to save the evaluation results')
    args = parser.parse_args()

    pred_path = Path(args.pred_path)
    gt_path = Path(args.gt_path)
    # there might be more test imgs than GTs
    gts = [f for f in gt_path.glob('*.png')]
    metrics = [DiceMetric(), MeanIoU()]

    # iterate over the ground truths
    for gt in gts:
        # get the corresponding prediction
        pred = pred_path / gt.name
        print(pred)
        # compute the metrics
        # for metric in metrics:
        #     value = compute_metrics(pred, gt, metric)
        #     print(f'{metric.__class__.__name__} for {gt.name}: {value}')


if __name__ == '__main__':
    main()

