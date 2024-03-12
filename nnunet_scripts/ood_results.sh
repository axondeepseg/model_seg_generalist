#!/bin/bash
OOD_PATH=$1
NNUNET_PATH=$2

shift 2
models=("$@")

# print the structure of the OOD dataset
echo -e "\n############## Printing the structure of the OOD dataset. ##############\n"
tree $OOD_PATH

# Get all second level directories
second_level_dirs=()
for dir in "$OOD_PATH"/*; do
    if [ -d "$dir" ]; then
        for sub_dir in "$dir"/*; do
            if [ -d "$sub_dir" ]; then
                second_level_dirs+=("$sub_dir")
            fi
        done
    fi
done

# Convert all images to grayscale (Expected by nnunet models)
echo -e "\n############## Converting all images to grayscale. ##############\n"
for dir in ${second_level_dirs[@]}; do
    python utils/make_gray_scale.py $dir
done

echo -e "\n############## Running OOD inference and evaluation. ##############\n"
for model in ${models[@]}; do
    for dataset in ${second_level_dirs[@]}; do
        echo -e "\n\nRun inference with ${NNUNET_PATH}/nnUNet_results/${model} model on $dataset\n\n"
        model_name=${model#Dataset???_}

        python nnunet_scripts/run_inference.py  --path-dataset $dataset \
                                                --path-out ${dataset}/segmentations/ensemble/${model_name}_model_best_checkpoints_inference \
                                                --path-model ${NNUNET_PATH}/nnUNet_results/${model}/nnUNetTrainer__nnUNetPlans__2d/ \
                                                --use-gpu \
                                                --use-best-checkpoint

        if [ -d "$dataset/labels" ]; then
            python nnunet_scripts/run_evaluation.py --pred_path ${dataset}/segmentations/ensemble/${model_name}_model_best_checkpoints_inference \
                                                    --gt_path ${dataset}/labels \
                                                    --output_fname ${dataset}/segmentations/ensemble/scores/${model_name}_model_best_checkpoints_scores \
                                                    --pred_suffix _0000
        else
            echo "Skipping evaluation of $dataset as it does not contain a 'labels' subdirectory."
        fi
                                    

        python utils/make_segmentations_visible.py  ${dataset}/segmentations/ensemble/${model_name}_model_best_checkpoints_inference \
                                                    ${dataset}/segmentations/ensemble/${model_name}_model_best_checkpoints_inference

        # Run inference and evaluation for each fold
        for fold in {0..4}; do
            echo -e "\nProcessing fold $fold\n"
            python nnunet_scripts/run_inference.py  --path-dataset $dataset \
                                                    --path-out ${dataset}/segmentations/fold_${fold}/${model_name}_model_best_checkpoints_inference \
                                                    --path-model ${NNUNET_PATH}/nnUNet_results/${model}/nnUNetTrainer__nnUNetPlans__2d/ \
                                                    --use-gpu \
                                                    --folds $fold \
                                                    --use-best-checkpoint

            if [ -d "$dataset/labels" ]; then
                python nnunet_scripts/run_evaluation.py --pred_path ${dataset}/segmentations/fold_${fold}/${model_name}_model_best_checkpoints_inference \
                                                        --gt_path ${dataset}/labels \
                                                        --output_fname ${dataset}/segmentations/fold_${fold}/scores/${model_name}_model_best_checkpoints_scores \
                                                        --pred_suffix _0000
            fi

            python utils/make_segmentations_visible.py  ${dataset}/segmentations/fold_${fold}/${model_name}_model_best_checkpoints_inference \
                                                        ${dataset}/segmentations/fold_${fold}/${model_name}_model_best_checkpoints_inference
        done
    done
done
