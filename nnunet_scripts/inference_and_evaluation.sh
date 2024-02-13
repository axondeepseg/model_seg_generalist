
# This script processes a list of datasets. The first argument should be the path to the datasets.
# The remaining arguments should be the names of the datasets to process.
NNUNET_PATH=$1
DATASET_NAME="Dataset444_AGG"

if [ ! -d "${NNUNET_PATH}/nnUNet_results/${DATASET_NAME}/scores" ]; then
  echo "Making directory ${NNUNET_PATH}/nnUNet_results/${DATASET_NAME}/scores" 
  mkdir "${NNUNET_PATH}/nnUNet_results/${DATASET_NAME}/scores"
fi

shift
for dataset in "$@"; do
  echo -e "\n\nRun inference with $NNUNET_PATH/$dataset model on $DATASET_NAME dataset\n\n"
  dataset_name=${dataset#Dataset???_}

  python nnunet_scripts/run_inference.py  --path-dataset ${NNUNET_PATH}/nnUNet_raw/${DATASET_NAME}/imagesTs \
                                          --path-out ${NNUNET_PATH}/nnUNet_results/${DATASET_NAME}/${dataset_name}_model_best_checkpoints_inference \
                                          --path-model ~/data/nnunet_all/nnUNet_results/${dataset}/nnUNetTrainer__nnUNetPlans__2d/ \
                                          --use-gpu \
                                          --use-best-checkpoint

  python nnunet_scripts/run_evaluation.py --pred_path ${NNUNET_PATH}/nnUNet_results/${DATASET_NAME}/${dataset_name}_model_best_checkpoints_inference \
                                          --mapping_path ${NNUNET_PATH}/nnUNet_raw/${DATASET_NAME}/fname_mapping.json \
                                          --gt_path ${NNUNET_PATH}/test_labels \
                                          --output_fname ${NNUNET_PATH}/nnUNet_results/${DATASET_NAME}/scores/${dataset_name}_model_best_checkpoints_scores \
                                          --pred_suffix _0000

done