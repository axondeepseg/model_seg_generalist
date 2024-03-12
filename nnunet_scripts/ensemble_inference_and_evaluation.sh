# The first argument should be the path to the datasets.
# The remaining arguments should be the names of the datasets to process.
NNUNET_PATH=$1
DATASET_NAME="Dataset444_AGG"

shift
for dataset in "$@"; do
  echo -e "\n\nRun inference with ${NNUNET_PATH}/nnUNet_results/${dataset} model on $DATASET_NAME dataset\n\n"
  dataset_name=${dataset#Dataset???_}

  # Run inference and save the probabilities
  python nnunet_scripts/run_inference.py  --path-dataset ${NNUNET_PATH}/nnUNet_raw/${DATASET_NAME}/imagesTs \
                                          --path-out ${NNUNET_PATH}/nnUNet_results/${DATASET_NAME}/ensemble_models/${dataset_name}_model_best_checkpoints_inference_with_probabilities \
                                          --path-model ${NNUNET_PATH}/nnUNet_results/${dataset}/nnUNetTrainer__nnUNetPlans__2d/ \
                                          --use-gpu \
                                          --use-best-checkpoint \
                                          --save-probabilities 
done

folders=()
ensemble_name="ensemble"
for dataset in "$@"; do
  dataset_name=${dataset#Dataset???_}
  folders+=("${NNUNET_PATH}/nnUNet_results/${DATASET_NAME}/ensemble_models/${dataset_name}_model_best_checkpoints_inference_with_probabilities")
  ensemble_name+="_${dataset_name}"
done

# Ensemble the inference results with nnUNetv2_ensemble
nnUNetv2_ensemble -i ${folders[@]} -o ${NNUNET_PATH}/nnUNet_results/${DATASET_NAME}/ensemble_models/${ensemble_name}_model_best_checkpoints_inference 

# Evaluate the ensemble model
python nnunet_scripts/run_evaluation.py --pred_path ${NNUNET_PATH}/nnUNet_results/${DATASET_NAME}/ensemble_models/${ensemble_name}_model_best_checkpoints_inference \
                                        --mapping_path ${NNUNET_PATH}/nnUNet_raw/${DATASET_NAME}/fname_mapping.json \
                                        --gt_path ${NNUNET_PATH}/test_labels \
                                        --output_fname ${NNUNET_PATH}/nnUNet_results/${DATASET_NAME}/ensemble_models/scores/${ensemble_name}_model_best_checkpoints_scores \
                                        --pred_suffix _0000