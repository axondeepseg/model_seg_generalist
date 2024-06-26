# model_seg_generalist

## How to train
This model can be trained on multiple datasets. The preprocessing script expects the datasets, already in nnunet format, to be located in a specific directory (the directory containing `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`). The indices of the datasets to be used for training must be passed as arguments to the script.

The datasets could be any of the following:

- TEM dataset
- SEM dataset
- BF dataset (default)
- BF dataset (wakehealth, human)
- BF dataset (VCU, rabbit)

First, run the aggregation script with the required arguments. The script expects the following arguments:

- `--nnunet_dir`: The path to the directory containing the datasets.
- `--dataset_ids`: A list of indices of the datasets to be used for training. If no indices are provided, all datasets in the directory will be used.
- `--name`: (Optional) The name you want to assign to the aggregated dataset. Defaults to 'Dataset444_AGG'.
- `--description`: (Optional) A description for the aggregated dataset. Defaults to 'Aggregated dataset from all source domains'.
- `--k`: (Optional) The number of folds for cross-validation. Defaults to 5.

Here is an example command to run the script:
```
python nnunet_scripts/aggregate_data.py --nnunet_dir path_to_directory_containing_dsets --dataset_ids <dataset_index_1> <dataset_index_2> ... <dataset_index_n> --name MyAggregatedDataset --description "Aggregated dataset for my experiment" --k 5
```
Replace `<dataset_index_1> <dataset_index_2> ... <dataset_index_n>` with the indices of the datasets you want to use for training.

This will create a new nnunet dataset. We can then run the initial setup, 
move the manual split in the preprocessed folder and start training:
```bash
source ./nnunet_scripts/setup_nnunet.sh NNUNET_DIR
./nnunet_scripts/train_nnunet.sh 444 AGG <GPU_ID> <FOLD_1> <FOLD_2> ... <FOLD_k>
```
To parallelize the execution of the training script for faster processing, you can run multiple instances of the script simultaneously, each handling a different fold for cross-validation. This is particularly useful when you have access to a machine with multiple GPUs. Here's an example command that demonstrates how to run training on 5 folds in parallel:

```bash
./nnunet_scripts/train_nnunet.sh 444 AGG <GPU_ID> 0 &
./nnunet_scripts/train_nnunet.sh 444 AGG <GPU_ID> 1 & 
./nnunet_scripts/train_nnunet.sh 444 AGG <GPU_ID> 2 & 
./nnunet_scripts/train_nnunet.sh 444 AGG <GPU_ID> 3 & 
./nnunet_scripts/train_nnunet.sh 444 AGG <GPU_ID> 4 & 
```


## Setting Up Conda Environment

To set up the environment and run the scripts, follow these steps:

1. Create a new conda environment:
```bash
conda create --name generalist_seg
```
2. Activate the environment:
```bash
conda activate generalist_seg
```
3. Install PyTorch, torchvision, and torchaudio. For NeuroPoly lab members using the GPU servers, use the following command:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
For others, please refer to the PyTorch installation guide at https://pytorch.org/get-started/locally/ to get the appropriate command for your system.

4. Update the environment with the remaining dependencies:
```bash
conda env update --file environment.yaml
```


## Inference

After training the model, you can perform inference using the following command:
```bash
python nnunet_scripts/run_inference.py --path-dataset ${nnUNet_raw}/Dataset<FORMATTED_DATASET_ID>_<DATASET_NAME>/imagesTs --path-out <WHERE/TO/SAVE/RESULTS> --path-model ${nnUNet_results}/Dataset<FORMATTED_DATASET_ID>_<DATASET_NAME>/nnUNetTrainer__nnUNetPlans__2d/ --use-gpu --use-best-checkpoint
```
The `--use-best-checkpoint` flag is optional. If used, the model will use the best checkpoints for inference. If not used, the model will use the latest checkpoints. Based on empirical results, using the `--use-best-checkpoint` flag is recommended.

Note: `<FORMATTED_DATASET_ID>` should be a three-digit number where 1 would become 001 and 23 would become 023.

## Replicating Experiments

To replicate the inference experiments, execute the following script:

```bash
source ./nnunet_scripts/inference_and_evaluation.sh ${NNUNET_DIR} <DATASET_1> <DATASET_2> <DATASET_3> ... <DATASET_N>
```

For instance, to run the script with specific datasets, use the command below:

```bash
source ./nnunet_scripts/inference_and_evaluation.sh ${NNUNET_DIR} Dataset002_SEM Dataset003_TEM Dataset004_BF_RAT Dataset005_wakehealth Dataset006_BF_VCU Dataset444_AGG
```
In addition to the individual inference and evaluation scripts, there is an "ensemble_inference_and_evaluation.sh" script available. This script performs ensemble inferences using all the models listed and then evaluates the ensemble model. The arguments for this script are similar to the ones mentioned above, except `<DATASET_K>` represents all the models being ensembled.

To use the ensemble script, execute the following command:
```bash
source ./nnunet_scripts/ensemble_inference_and_evaluation.sh ${NNUNET_DIR} <DATASET_1> <DATASET_2> <DATASET_3> ... <DATASET_N>
```

To replicate out of distribution experiments (OOD), you can use the following script:

```bash
source ./nnunet_scripts/ood_results.sh <PATH_TO_OOD_DATASET> ${RESULTS_DIR}/nnUNet_results <DATASET_1> <DATASET_2> <DATASET_3> ... <DATASET_N>
```

Ensure the OOD dataset adheres to the following structure prior to executing the script:

```
├── <MODALITY 1>
│   ├── some species
│       ├── image_0000.png
│       └── (optional) labels
│            └── image.png
|   ...
│   └── another species
│       ├── image_0000.png
│       └── (optional) labels
│            └── image.png
└── <MODALITY N>
    ...
```

For instance:

```   
├── BF
│   └── cat
│       └── CAT_0000.png
├── SEM
│   ├── dog
│   │   └── DOG_0000.png
│   └── human
│       ├── AGG_203_0000.png
│       └── labels
│            └── AGG_203.png
└── TEM
    └── macaque
        ├── labels
              ├── MACAQUE_000_0000.png
              ├── MACAQUE_001_0000.png
              ├── MACAQUE_002_0000.png
              ├── MACAQUE_003_0000.png
              ├── MACAQUE_004_0000.png
             ...
        ├── MACAQUE_000_0000.png
        ├── MACAQUE_001_0000.png
        ├── MACAQUE_002_0000.png
        ├── MACAQUE_003_0000.png
        ├── MACAQUE_004_0000.png
       ...
```
## Authors

- Armand Collin
- Arthur Boschet

