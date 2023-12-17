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

    fname_mapping = {'images_tr': [], 'images_ts': []}
    new_index = 0
    # rename every file and log mapping
    for dataset in data_dict.keys():
        for im_tr in data_dict[dataset]['imagesTr']:
            new_fname = f'AGG_{new_index:03d}_0000.png'
            old_fname = str(im_tr.name)
            fname_mapping['images_tr'].append([old_fname, new_fname])
            new_index += 1
    fname_mapping['numTraining'] = len(fname_mapping['images_tr'])
    for dataset in data_dict.keys():
        for im_ts in data_dict[dataset]['imagesTs']:
            new_fname = f'AGG_{new_index:03d}_0000.png'
            old_fname = str(im_ts.name)
            fname_mapping['images_ts'].append([old_fname, new_fname])
            new_index += 1
    fname_mapping['numTest'] = len(fname_mapping['images_ts'])
    # save mapping
    with open(output_dir / 'fname_mapping.json', 'w') as f:
        json.dump(fname_mapping, f)



if __name__ == "__main__":
    main()
