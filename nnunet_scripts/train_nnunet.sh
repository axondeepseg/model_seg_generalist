#!/bin/bash
#
# Training nnUNetv2 on multiple folds
#
# NOTE: This is a template script, modify it as needed
#
# Modified from the script by Naga Karthik and Jan Valosek
# located in: https://github.com/ivadomed/utilities/blob/main/scripts/run_nnunet.sh
#

# This script is used to train nnUNet on multiple folds.
# It takes four arguments:
# 1. DATASET_ID: The ID of the dataset to be used for training. This should be an integer.
# 2. DATASET_NAME: The name of the dataset. This will be used to form the full dataset name in the format "DatasetNUM_DATASET_NAME".
# 3. DEVICE: The device to be used for training. This could be a GPU device ID or 'cpu' for CPU.
# 4. FOLDS: The folds to be used for training. This should be a space-separated list of integers.
# Example usage: ./nnunet_scripts/train_nnunet.sh 1 SEM 0 0 1 2 3 4

config="2d"                     
dataset_id=$1   
dataset_name="Dataset$(printf "%03d" $dataset_id)_$2"     
nnunet_trainer="nnUNetTrainer"
DEVICE=$3 # No default device

# Check if the required arguments (dataset_id, dataset_name, folds) are provided
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 DATASET_ID DATASET_NAME DEVICE FOLDS"
    exit 1
fi

# Convert the argument to an array of folds
folds=("${@:4}") # Skip the first three arguments (DATASET_ID, DATASET_NAME, DEVICE)

for fold in ${folds[@]}; do
    echo "-------------------------------------------"
    echo "Training on Fold $fold"
    echo "-------------------------------------------"

    # Check if CUDA is enabled
    if [[ $DEVICE =~ ^[0-9]+$ ]]; then
        # training
        CUDA_VISIBLE_DEVICES=$DEVICE nnUNetv2_train ${dataset_id} ${config} ${fold} -tr ${nnunet_trainer}

        echo ""
        echo "-------------------------------------------"
        echo "Training completed, Testing on Fold $fold"
        echo "-------------------------------------------"

        # inference
        CUDA_VISIBLE_DEVICES=$DEVICE nnUNetv2_predict -i ${nnUNet_raw}/${dataset_name}/imagesTs -tr ${nnunet_trainer} -o ${nnUNet_results}/${nnunet_trainer}__nnUNetPlans__${config}/fold_${fold}/test -d ${dataset_id} -f ${fold} -c ${config}
    else
        # training
        nnUNetv2_train ${dataset_id} ${config} ${fold} -tr ${nnunet_trainer} -device ${DEVICE}

        echo ""
        echo "-------------------------------------------"
        echo "Training completed, Testing on Fold $fold"
        echo "-------------------------------------------"

        # inference
        nnUNetv2_predict -i ${nnUNet_raw}/${dataset_name}/imagesTs -tr ${nnunet_trainer} -o ${nnUNet_results}/${nnunet_trainer}__nnUNetPlans__${config}/fold_${fold}/test -d ${dataset_id} -f ${fold} -c ${config} -device ${DEVICE}
    fi

    echo ""
    echo "-------------------------------------------"
    echo " Inference completed on Fold $fold"
    echo "-------------------------------------------"

done