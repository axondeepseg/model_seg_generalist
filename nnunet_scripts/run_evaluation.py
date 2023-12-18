"""
This script is used to run evaluation on a generalist model trained in the 
nnUNet framework. It will compute the metrics for every test image and display 
results for every dataset separately.
Author: Armand Collin
"""

import argparse
from pathlib import Path
from monai.metrics import DiceMetric

def main():
    parser = argparse.ArgumentParser(description='Run evaluation on a generalist model')
    parser.add_argument('-p', '--pred_path', type=str, help='Path to the predictions folder')
    parser.add_argument('-m', '--mapping_path', type=str, help='Path to the filename mapping JSON file')
    parser.add_argument('-g', '--gt_path', type=str, help='Path to the GT folder')
    parser.add_argument('-o', '--output_path', type=str, help='Path to save the evaluation results')
    args = parser.parse_args()

    pred_path = Path(args.pred_path)
    gt_path = Path(args.gt_path)
    preds = [x for x in pred_path.glob('AGG_*.png')]


if __name__ == '__main__':
    main()

