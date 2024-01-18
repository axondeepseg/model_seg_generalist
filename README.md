# model_seg_generalist

## How to train
This model is trained on 5 datasets. The preprocessing script expects the following 5 datasets, already in nnunet format:

- TEM dataset
- SEM dataset
- BF dataset (default)
- BF dataset (wakehealth, human)
- BF dataset (VCU, rabbit)

First, run the aggregation script:
```
python nnunet_scripts/aggregate_data.py -i path_to_directory_containing_dsets -o .
```

This will create a new nnunet dataset. We can then run the initial setup, 
move the manual split in the preprocessed folder and start training:
```
./nnunet_scripts/setup_nnunet.sh ABSOLUTE_PATH_TO_CWD
cp final_splits.json nnUNet_preprocessed/Dataset444_AGG/
./nnunet_scripts/train_3fold.sh
```


## Inference

After training the model, you can perform inference using the following command:
```bash
python nnunet_scripts/run_inference.py --path-dataset ${nnUNet_raw}/Dataset<FORMATTED_DATASET_ID>_<DATASET_NAME>/imagesTs --path-out <WHERE/TO/SAVE/RESULTS> --path-model ${nnUNet_results}/Dataset<FORMATTED_DATASET_ID>_<DATASET_NAME>/nnUNetTrainer__nnUNetPlans__2d/ --use-gpu --use-best-checkpoint
```
The `--use-best-checkpoint` flag is optional. If used, the model will use the best checkpoints for inference. If not used, the model will use the latest checkpoints. Based on empirical results, using the `--use-best-checkpoint` flag is recommended.

Note: `<FORMATTED_DATASET_ID>` should be a three-digit number where 1 would become 001 and 23 would become 023.

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
