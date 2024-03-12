
# This script processes a list of datasets. The first argument should be the path to the datasets.
# The remaining arguments should be the names of the datasets to process.
NNUNET_PATH=$1
DATASET_NAME="Dataset444_AGG"

shift
for dataset in "$@"; do
  echo -e "\n\nRun inference with ${NNUNET_PATH}/nnUNet_results/${dataset} model on $DATASET_NAME dataset\n\n"
  dataset_name=${dataset#Dataset???_}

  python nnunet_scripts/run_inference.py  --path-dataset ${NNUNET_PATH}/nnUNet_raw/${DATASET_NAME}/imagesTs \
                                          --path-out ${NNUNET_PATH}/nnUNet_results/${DATASET_NAME}/ensemble_models/${dataset_name}_model_best_checkpoints_inference \
                                          --path-model ${NNUNET_PATH}/nnUNet_results/${dataset}/nnUNetTrainer__nnUNetPlans__2d/ \
                                          --use-gpu \
                                          --use-best-checkpoint

  python nnunet_scripts/run_evaluation.py --pred_path ${NNUNET_PATH}/nnUNet_results/${DATASET_NAME}/ensemble_models/${dataset_name}_model_best_checkpoints_inference \
                                          --mapping_path ${NNUNET_PATH}/nnUNet_raw/${DATASET_NAME}/fname_mapping.json \
                                          --gt_path ${NNUNET_PATH}/test_labels \
                                          --output_fname ${NNUNET_PATH}/nnUNet_results/${DATASET_NAME}/ensemble_models/scores/${dataset_name}_model_best_checkpoints_scores \
                                          --pred_suffix _0000

  # Run inference and evaluation for each fold
  for fold in {0..4}; do
    echo -e "\nProcessing fold $fold\n"
    python nnunet_scripts/run_inference.py  --path-dataset ${NNUNET_PATH}/nnUNet_raw/${DATASET_NAME}/imagesTs \
                                            --path-out ${NNUNET_PATH}/nnUNet_results/${DATASET_NAME}/fold_{$fold}/${dataset_name}_model_best_checkpoints_inference \
                                            --path-model ${NNUNET_PATH}/nnUNet_results/${dataset}/nnUNetTrainer__nnUNetPlans__2d/ \
                                            --folds $fold \
                                            --use-gpu \
                                            --use-best-checkpoint

    python nnunet_scripts/run_evaluation.py --pred_path ${NNUNET_PATH}/nnUNet_results/${DATASET_NAME}/fold_{$fold}/${dataset_name}_model_best_checkpoints_inference \
                                            --mapping_path ${NNUNET_PATH}/nnUNet_raw/${DATASET_NAME}/fname_mapping.json \
                                            --gt_path ${NNUNET_PATH}/test_labels \
                                            --output_fname ${NNUNET_PATH}/nnUNet_results/${DATASET_NAME}/fold_{$fold}/scores/${dataset_name}_model_best_checkpoints_scores \
                                            --pred_suffix _0000
  done

  # Run inference and evaluation for fold_all
  echo -e "\nProcessing fold_all\n"
  python nnunet_scripts/run_inference.py  --path-dataset ${NNUNET_PATH}/nnUNet_raw/${DATASET_NAME}/imagesTs \
                                          --path-out ${NNUNET_PATH}/nnUNet_results/${DATASET_NAME}/fold_all/${dataset_name}_model_best_checkpoints_inference \
                                          --path-model ${NNUNET_PATH}/nnUNet_results/${dataset}/nnUNetTrainer__nnUNetPlans__2d/ \
                                          --fold_all \
                                          --use-gpu \
                                          --use-best-checkpoint

  python nnunet_scripts/run_evaluation.py --pred_path ${NNUNET_PATH}/nnUNet_results/${DATASET_NAME}/fold_all/${dataset_name}_model_best_checkpoints_inference \
                                          --mapping_path ${NNUNET_PATH}/nnUNet_raw/${DATASET_NAME}/fname_mapping.json \
                                          --gt_path ${NNUNET_PATH}/test_labels \
                                          --output_fname ${NNUNET_PATH}/nnUNet_results/${DATASET_NAME}/fold_all/scores/${dataset_name}_model_best_checkpoints_scores \
                                          --pred_suffix _0000
done