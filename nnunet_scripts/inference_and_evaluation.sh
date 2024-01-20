
# This script processes a list of datasets. The first argument should be the path to the datasets.
# The remaining arguments should be the names of the datasets to process.

if [ ! -d "${RESULTS_DIR}/nnUNet_results/Dataset444_AGG/scores" ]; then
  echo "hello" 
  mkdir "${RESULTS_DIR}/nnUNet_results/Dataset444_AGG/scores"
fi



DATASET_PATH=$1
shift
for dataset in "$@"; do
  echo -e "\n\nRun inference on $DATASET_PATH/$dataset"

  dataset_name=${dataset#Dataset???_}

  python nnunet_scripts/run_inference.py  --path-dataset ${RESULTS_DIR}/nnUNet_raw/Dataset444_AGG/imagesTs \
                                          --path-out ${RESULTS_DIR}/nnUNet_results/Dataset444_AGG/${dataset_name}_model_best_checkpoints_inference \
                                          --path-model ~/data/nnunet_all/nnUNet_results/${dataset}/nnUNetTrainer__nnUNetPlans__2d/ \
                                          --use-gpu \
                                          --use-best-checkpoint

  python nnunet_scripts/run_evaluation.py --pred_path ${RESULTS_DIR}/nnUNet_results/Dataset444_AGG/${dataset_name}_model_best_checkpoints_inference \
                                          --mapping_path ${RESULTS_DIR}/fname_mapping.json \
                                          --gt_path ${RESULTS_DIR}/test_labels \
                                          --output_fname ${RESULTS_DIR}/nnUNet_results/Dataset444_AGG/scores/${dataset_name}_model_best_checkpoints_scores \
                                          --pred_suffix _0000

done