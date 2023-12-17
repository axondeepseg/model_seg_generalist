#!/bin/bash
# This script sets up the nnUNet environment and runs the preprocessing and dataset integrity verification

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 RESULTS_DIR [DATASET_ID] [DATASET_NAME]"
    exit 1
fi

RESULTS_DIR=$(realpath $1)
dataset_id=${2:-444}
dataset_name=${3:-"AGG"}

echo "-------------------------------------------------------"
echo "Converting dataset to nnUNetv2 format"
echo "-------------------------------------------------------"

# Set up the necessary environment variables
export nnUNet_raw="$RESULTS_DIR/nnUNet_raw"
export nnUNet_preprocessed="$RESULTS_DIR/nnUNet_preprocessed"
export nnUNet_results="$RESULTS_DIR/nnUNet_results"

echo "-------------------------------------------------------"
echo "Running preprocessing and verifying dataset integrity"
echo "-------------------------------------------------------"

nnUNetv2_plan_and_preprocess -d ${dataset_id} --verify_dataset_integrity -c "2d"