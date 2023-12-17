"""
This script aggregates data by processing images from all datasets and 
copying them into a new single nnunet formatted dataset. It also creates a 
3-fold cross-validation scheme manually to ensure the splits contain data from 
all domains.
Author: Armand Collin
"""

import argparse
import json
from pathlib import Path
from sklearn.model_selection import KFold

def create_splits(dataset, n_splits=3):
    '''Creates n_splits train-val splits for the given dataset. Please 
    note that the splits might not have exactly the same length.'''
    kf = KFold(n_splits=n_splits)
    splits = []
    for train_index, val_index in kf.split(dataset):
        train = [dataset[i] for i in train_index]
        val = [dataset[i] for i in val_index]
        splits.append((train, val))
    return splits

def map_images(datasets, image_type, current_index):
    fname_mapping = {}
    for dataset in datasets:
        images = [im for im in (dataset / image_type).glob('*.png')]
        for image in images:
            new_fname = f'AGG_{current_index:03d}_0000.png'
            old_fname = str(image.name)
            fname_mapping[old_fname] = new_fname
            current_index += 1
    return fname_mapping, current_index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Where the datasets are located")
    parser.add_argument("-o", "--output", help="Where to save the aggregated dataset")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    datasets = [x for x in input_dir.glob('Dataset*')]
    data_dict = {}
    # Iterate over all datasets
    for dataset in datasets:
        images_tr = [im for im in (dataset / 'imagesTr').glob('*.png')]
        images_ts = [im for im in (dataset / 'imagesTs').glob('*.png')]
        data_dict[str(dataset)] = {
            'imagesTr': images_tr,
            'imagesTs': images_ts,
            'numTraining': len(images_tr),
            'numTest': len(images_ts)
        }

    new_index = 0
    fname_mapping_tr, new_index = map_images(datasets, 'imagesTr', new_index)
    fname_mapping_ts, new_index = map_images(datasets, 'imagesTs', new_index)
    fname_mapping = {
        'images_tr': fname_mapping_tr,
        'images_ts': fname_mapping_ts,
        'numTraining': len(fname_mapping_tr),
        'numTest': len(fname_mapping_ts)
    }

    # save mapping
    with open(output_dir / 'fname_mapping.json', 'w') as f:
        json.dump(fname_mapping, f)

    # create the 3-fold train-val splits for every dataset:
    train_val_splits = {str(dataset): [] for dataset in datasets}
    for dataset_name in datasets:
        dataset = str(dataset_name)
        images_tr = data_dict[dataset]['imagesTr']
        images_tr = [str(i.name) for i in images_tr]
        train_val_splits[dataset] = create_splits(images_tr, n_splits=3)

    # convert splits from source domains to target domain
    target_splits = [{'train': [], 'val': []} for i in range(3)]
    for fold in range(3):
        for dataset in datasets:
            current_split = train_val_splits[str(dataset)][fold]
            train, val = current_split
            target_train = [fname_mapping['images_tr'][im].replace('_0000.png', '') for im in train]
            target_val = [fname_mapping['images_tr'][im].replace('_0000.png', '') for im in val]
            target_splits[fold]['train'].extend(target_train)
            target_splits[fold]['val'].extend(target_val)
    # save splits
    with open(output_dir / 'final_splits.json', 'w') as f:
        json.dump(target_splits, f)

    # finally, copy files into the new aggregated dataset
    output_data_path = output_dir / 'nnUNet_raw' / 'Dataset444_AGG'
    output_data_path.mkdir(exist_ok=False)
    (output_data_path / 'imagesTr').mkdir(exist_ok=False)
    (output_data_path / 'labelsTr').mkdir(exist_ok=False)
    (output_data_path / 'imagesTs').mkdir(exist_ok=False)
    for dataset in datasets:
        dataset_name = str(dataset)
        images_tr = data_dict[dataset_name]['imagesTr']
        images_ts = data_dict[dataset_name]['imagesTs']
        for image in images_tr:
            new_fname = fname_mapping['images_tr'][image.name]
            image_path_target = output_data_path / 'imagesTr' / new_fname
            image_path_target.symlink_to(image)
            # also symlink the corresponding label
            label_fname = image.name.replace('_0000.png', '.png')
            label = image.parent.parent / 'labelsTr' / label_fname
            new_label_fname = new_fname.replace('_0000.png', '.png')
            label_path_target = output_data_path / 'labelsTr' / new_label_fname
            label_path_target.symlink_to(label)
        for image in images_ts:
            new_fname = fname_mapping['images_ts'][image.name]
            image_path = output_data_path / 'imagesTs' / new_fname
            image_path.symlink_to(image)
    # create dataset.json file
    dataset_json = {
        'name': 'Dataset444_AGG',
        'description': 'Aggregated dataset from all source domains',
        "labels": {
            "background": 0,
            "myelin": 1,
            "axon": 2
        },
        "channel_names": {
            "0": "rescale_to_0_1"
        },
        "numTraining": fname_mapping['numTraining'],
        "numTest": fname_mapping['numTest'],
        "file_ending": ".png"
    }
    with open(output_data_path / 'dataset.json', 'w') as f:
        json.dump(dataset_json, f)


if __name__ == "__main__":
    main()
