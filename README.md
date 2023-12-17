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