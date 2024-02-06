"""
This script aggregates data by processing images from all datasets and 
copying them into a new single nnunet formatted dataset. It also creates a 
k-fold cross-validation scheme manually to ensure the splits contain data from 
all domains.
Author: Armand Collin
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Literal

from sklearn.model_selection import KFold


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.ArgumentParser: Argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nnunet_dir",
        help="Path to nnunet directory, where the existing datasets are stored and where the new dataset will be created.",
    )
    parser.add_argument(
        "--dataset_ids",
        nargs="*",
        default=None,
        help="List of dataset indices to be combined. Default: all datasets",
    )
    parser.add_argument(
        "--name",
        default="Dataset444_AGG",
        help="Name of the aggregated dataset. Default: 'Dataset444_AGG'",
    )
    parser.add_argument(
        "--description",
        default="Aggregated dataset from all source domains",
        help="Description of the aggregated dataset. Default: 'Aggregated dataset from all source domains'",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of folds for cross-validation. Default: 5",
    )
    return parser.parse_args()


def create_splits(dataset: List[str], n_splits: int = 5):
    """
    Creates n_splits train-val splits for the given dataset. Please
    note that the splits might not have exactly the same length.

    Parameters
    ----------
    dataset : List[str]
        List of image names in the dataset.
    n_splits : int, optional
        Number of splits to create, by default 5

    Returns
    -------
    splits : List[Tuple[List[str], List[str]]]
        List of train-val splits for each fold.
    """
    kf = KFold(n_splits=n_splits)
    splits = []
    for train_index, val_index in kf.split(dataset):
        train = [dataset[i] for i in train_index]
        val = [dataset[i] for i in val_index]
        splits.append((train, val))
    return splits


def map_images(
    datasets: List[str],
    image_type: Literal["imagesTr", "imagesTs"],
    current_index: int,
    acronym: str = "AGG",
):
    """
    Creates a mapping between the original filenames and the new filenames

    Parameters
    ----------
    datasets : List[str]
        List of datasets to aggregate.
    image_type : Literal["imagesTr", "imagesTs"]
        Type of images to aggregate. Either training or test images.
    current_index : int
        Current index of the image to aggregate.
    acronym : str, optional
        Acronym of the aggregated dataset which is added in front of the image names, by default "AGG".
    """
    fname_mapping = {}
    for dataset in datasets:
        images = [im for im in (dataset / image_type).glob("*.png")]
        for image in images:
            new_fname = f"{acronym}_{current_index:03d}_0000.png"
            old_fname = str(image.name)
            fname_mapping[old_fname] = new_fname
            current_index += 1
    return fname_mapping, current_index


def main():
    # Parse command line arguments
    args = parse_args()

    # Number of folds for cross-validation
    k = args.k

    # Directories for the aggregated dataset
    nnunet_dir = Path(args.nnunet_dir)
    nnunet_raw_dir = nnunet_dir / "nnUNet_raw"

    raw_dataset_path = nnunet_dir / "nnUNet_raw" / args.name
    raw_dataset_path.mkdir(exist_ok=False)

    preprocessed_dataset_path = nnunet_dir / "nnUNet_preprocessed" / args.name
    preprocessed_dataset_path.mkdir(exist_ok=False)

    # Convert dataset_ids to string of k characters each by padding zeros at the front
    if args.dataset_ids is not None:
        datasets = [f"Dataset{int(id):03d}" for id in args.dataset_ids]

    if args.dataset_ids is not None:
        # filter datasets
        datasets = [
            x for x in nnunet_raw_dir.glob("Dataset*") if x.name[:10] in datasets
        ]
    else:
        datasets = [x for x in nnunet_raw_dir.glob("Dataset*")]

    # create a dictionary with all the paths of each dataset
    data_dict = {}
    # Iterate over all datasets
    for dataset in datasets:
        images_tr = [im for im in (dataset / "imagesTr").glob("*.png")]
        images_ts = [im for im in (dataset / "imagesTs").glob("*.png")]
        data_dict[str(dataset)] = {
            "imagesTr": images_tr,
            "imagesTs": images_ts,
            "numTraining": len(images_tr),
            "numTest": len(images_ts),
        }

    # Create a mapping between the original filenames and the new filenames
    dataset_acronym = args.name[11:]
    new_index = 0
    fname_mapping_tr, new_index = map_images(
        datasets, "imagesTr", new_index, acronym=dataset_acronym
    )
    fname_mapping_ts, new_index = map_images(
        datasets, "imagesTs", new_index, acronym=dataset_acronym
    )
    fname_mapping = {
        "images_tr": fname_mapping_tr,
        "images_ts": fname_mapping_ts,
        "numTraining": len(fname_mapping_tr),
        "numTest": len(fname_mapping_ts),
    }

    # save mapping
    with open(raw_dataset_path / "fname_mapping.json", "w") as f:
        json.dump(fname_mapping, f)

    # create the k-fold train-val splits for every dataset:
    train_val_splits = {str(dataset): [] for dataset in datasets}
    for dataset_name in datasets:
        dataset = str(dataset_name)
        images_tr = data_dict[dataset]["imagesTr"]
        images_tr = [str(i.name) for i in images_tr]
        train_val_splits[dataset] = create_splits(images_tr, n_splits=k)

    # convert splits from source domains to target domain
    target_splits = [{"train": [], "val": []} for i in range(k)]
    for fold in range(k):
        for i, dataset in enumerate(datasets):
            # when combining the splits together, we want to make sure that there is approximately the same number of images in each validation set
            if i % 2 == 0:
                current_split = train_val_splits[str(dataset)][fold]
            else:
                current_split = train_val_splits[str(dataset)][k - fold - 1]
            train, val = current_split
            target_train = [
                fname_mapping["images_tr"][im].replace("_0000.png", "") for im in train
            ]
            target_val = [
                fname_mapping["images_tr"][im].replace("_0000.png", "") for im in val
            ]
            target_splits[fold]["train"].extend(target_train)
            target_splits[fold]["val"].extend(target_val)

    # save splits to file as described in nnunet documentation (https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/manual_data_splits.md)
    with open(preprocessed_dataset_path / "splits_final.json", "w") as f:
        json.dump(target_splits, f)

    # finally, copy files into the new aggregated dataset
    (raw_dataset_path / "imagesTr").mkdir(exist_ok=False)
    (raw_dataset_path / "labelsTr").mkdir(exist_ok=False)
    (raw_dataset_path / "imagesTs").mkdir(exist_ok=False)
    for dataset in datasets:
        dataset_name = str(dataset)
        images_tr = data_dict[dataset_name]["imagesTr"]
        images_ts = data_dict[dataset_name]["imagesTs"]

        # copy images and labels in the training sets
        for image in images_tr:
            # copy the image to the new dataset
            new_fname = fname_mapping["images_tr"][image.name]
            image_path_target = raw_dataset_path / "imagesTr" / new_fname
            shutil.copy(image, image_path_target)

            # also copy the corresponding label
            label_fname = image.name.replace("_0000.png", ".png")
            label = image.parent.parent / "labelsTr" / label_fname
            new_label_fname = new_fname.replace("_0000.png", ".png")
            label_path_target = raw_dataset_path / "labelsTr" / new_label_fname
            shutil.copy(label, label_path_target)

        # copy images in the test sets
        for image in images_ts:
            new_fname = fname_mapping["images_ts"][image.name]
            image_path = raw_dataset_path / "imagesTs" / new_fname
            shutil.copy(image, image_path)

    # create dataset.json file
    dataset_json = {
        "name": args.name,
        "description": args.description,
        "labels": {"background": 0, "myelin": 1, "axon": 2},
        "channel_names": {"0": "rescale_to_0_1"},
        "numTraining": fname_mapping["numTraining"],
        "numTest": fname_mapping["numTest"],
        "file_ending": ".png",
    }
    with open(raw_dataset_path / "dataset.json", "w") as f:
        json.dump(dataset_json, f)


if __name__ == "__main__":
    main()
