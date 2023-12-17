#!/bin/bash
#
# Training nnUNetv2 on multiple folds

config=2d
dataset_id=444
dataset_name=Dataset444_AGG
nnunet_trainer="nnUNetTrainer"

# Set up the necessary environment variables
export nnUNet_raw="`pwd`/nnUNet_raw"
export nnUNet_preprocessed="`pwd`/nnUNet_preprocessed"
export nnUNet_results="`pwd`/nnUNet_results"

# Select number of folds here
folds=(0 1 2)

for fold in ${folds[@]}; do
    echo "-------------------------------------------"
    echo "Training on Fold $fold"
    echo "-------------------------------------------"

    # training
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_train ${dataset_id} ${config} ${fold} -tr ${nnunet_trainer}

    echo ""
    echo "-------------------------------------------"
    echo "Training completed, Testing on Fold $fold"
    echo "-------------------------------------------"

    # inference
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i ${nnUNet_raw}/${dataset_name}/imagesTs -tr ${nnunet_trainer} -o ${nnUNet_results}/${nnunet_trainer}__nnUNetPlans__${config}/fold_${fold}/test -d ${dataset_id} -f ${fold} -c ${config}

    echo ""
    echo "-------------------------------------------"
    echo " Inference completed on Fold $fold"
    echo "-------------------------------------------"

done