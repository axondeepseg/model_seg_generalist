"""
This script runs inference on a whole dataset or on individual images using nnUNetv2.

Author: Naga Karthik
"""

import os
import argparse
import torch
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import join
import time

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

if 'nnUNet_raw' not in os.environ:
        os.environ['nnUNet_raw'] = 'UNDEFINED'
if 'nnUNet_results' not in os.environ:
        os.environ['nnUNet_results'] = 'UNDEFINED'
if 'nnUNet_preprocessed' not in os.environ:
        os.environ['nnUNet_preprocessed'] = 'UNDEFINED'

def get_parser() -> argparse.ArgumentParser:
    """
    Parse command line arguments.

    Returns:
        argparse.ArgumentParser: Argument parser.
    """
    parser = argparse.ArgumentParser(description='Segment images using nnUNet')
    parser.add_argument('--path-dataset', default=None, type=str,
                        help='Path to the test dataset folder. Use this argument only if you want '
                        'predict on a whole dataset.')
    parser.add_argument('--path-images', default=None, nargs='+', type=str,
                        help='List of images to segment. Use this argument only if you want '
                        'predict on a single image or list of invidiual images.')
    parser.add_argument('--path-out', help='Path to output directory.', required=True)
    parser.add_argument('--path-model', required=True, 
                        help='Path to the model directory. This folder should contain individual folders '
                        'like fold_0, fold_1, etc.',)
    parser.add_argument('--use-gpu', action='store_true', default=False,
                        help='Use GPU for inference. Default: False')
    parser.add_argument('--use-mirroring', action='store_true', default=False,
                        help='Use mirroring (test-time) augmentation for prediction. '
                        'NOTE: Inference takes a long time when this is enabled. Default: False')
    parser.add_argument('--use-best-checkpoint', action='store_true', default=False,
                        help='Use the best checkpoint (instead of the final checkpoint) for prediction. '
                        'NOTE: nnUNet by default uses the final checkpoint. Default: False')

    return parser


def splitext(fname: str) -> tuple:
    """
    Split a fname (folder/file + ext) into a folder/file and extension.

    Args:
        fname (str): File name.

    Returns:
        tuple: Folder/file and extension.
    """
    dir, filename = os.path.split(fname)
    for special_ext in ['.nii.gz', '.tar.gz']:
        if filename.endswith(special_ext):
            stem, ext = filename[:-len(special_ext)], special_ext
            return os.path.join(dir, stem), ext
    stem, ext = os.path.splitext(filename)
    return os.path.join(dir, stem), ext


def add_suffix(fname: str, suffix: str) -> str:
    """
    Add suffix between end of file name and extension.

    Args:
        fname (str): File name.
        suffix (str): Suffix.

    Returns:
        str: File name with suffix.
    """
    stem, ext = splitext(fname)
    return os.path.join(stem + suffix + ext)


def convert_filenames_to_nnunet_format(path_dataset: str) -> str:
    """
    Convert file names to nnunet format.

    Args:
        path_dataset (str): Path to the dataset.

    Returns:
        str: Path to the temporary folder.
    """
    path_tmp = os.path.join(os.path.dirname(path_dataset), 'tmp')
    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp, exist_ok=True)

    for f in os.listdir(path_dataset):
        if f.endswith('.nii.gz') or f.endswith('.png'):
            f = os.path.join(path_dataset, f)
            f_new = add_suffix(f, '_0000')
            os.system('cp {} {}'.format(f, os.path.join(path_tmp, os.path.basename(f_new))))

    return path_tmp


def main():
    """
    Main function to run the script.
    """
    parser = get_parser()
    args = parser.parse_args()

    if not os.path.exists(args.path_out):
        os.makedirs(args.path_out, exist_ok=True)

    if args.path_dataset is not None and args.path_images is not None:
        raise ValueError('You can only specify either --path-dataset or --path-images (not both). See --help for more info.')
    
    if args.path_dataset is not None:
        print('Found a dataset folder. Running inference on the whole dataset...')
        path_data_tmp = convert_filenames_to_nnunet_format(args.path_dataset)
        path_out = args.path_out

    elif args.path_images is not None:
        print(f'Found {len(args.path_images)} images. Running inference on them...')
        path_data_tmp = [[f] for f in args.path_images]

        path_out = []
        for f in args.path_images:
            fname = Path(f).name
            fname = fname.rstrip(''.join(Path(fname).suffixes))
            path_pred = os.path.join(args.path_out, add_suffix(fname, '_pred')) 
            path_out.append(path_pred)

    folds_avail = [int(f.split('_')[-1]) for f in os.listdir(args.path_model) if f.startswith('fold_')]

    print('Starting inference...')
    start = time.time()
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True if args.use_gpu else False,
        device=torch.device('cuda') if args.use_gpu else torch.device('cpu'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    print('Running inference on device: {}'.format(predictor.device))

    predictor.initialize_from_trained_model_folder(
        join(args.path_model),
        use_folds=folds_avail,
        checkpoint_name='checkpoint_final.pth' if not args.use_best_checkpoint else 'checkpoint_best.pth',
    )
    print('Model loaded successfully. Fetching test data...')

    predictor.predict_from_files(
        path_data_tmp, 
        path_out,
        save_probabilities=False, 
        overwrite=False,
        num_processes_preprocessing=2, 
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None, 
        num_parts=1, 
        part_id=0
    )
    end = time.time()
    print('Inference done.')

    print('Deleting the temporary folder...')
    os.system('rm -rf {}'.format(path_data_tmp))

    print('----------------------------------------------------')
    print('Results can be found in: {}'.format(args.path_out))
    print('----------------------------------------------------')

    print('Total time elapsed: {:.2f} seconds'.format(end - start))

if __name__ == '__main__':
    main()