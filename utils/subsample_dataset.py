import os
import shutil
import numpy as np
import argparse
from tqdm import tqdm  # Import tqdm for the progress bar

def subsample_dataset(src_dir, dest_dir, downsample_ratio=0.05):
    # Directories for images, depth images, and masks
    image_dir = os.path.join(src_dir, 'image')
    depth_dir = os.path.join(src_dir, 'depth')
    mask_dir = os.path.join(src_dir, 'mask')
    
    # Make sure the destination directories exist
    dest_image_dir = os.path.join(dest_dir, 'image')
    dest_depth_dir = os.path.join(dest_dir, 'depth')
    dest_mask_dir = os.path.join(dest_dir, 'mask')
    os.makedirs(dest_image_dir, exist_ok=True)
    os.makedirs(dest_depth_dir, exist_ok=True)
    os.makedirs(dest_mask_dir, exist_ok=True)
    
    # List all files in the image directory and remove file extensions
    files = [os.path.splitext(file)[0] for file in os.listdir(image_dir)]
    
    # Shuffle and downsample the file list
    np.random.shuffle(files)
    subset_size = int(len(files) * downsample_ratio)
    selected_files = files[:subset_size]
    
    # Copy the selected files to the destination directories, with progress bar
    for base_filename in tqdm(selected_files, desc='Copying files'):
        shutil.copy2(os.path.join(image_dir, base_filename + '.jpg'), dest_image_dir)
        shutil.copy2(os.path.join(depth_dir, base_filename + '.png'), dest_depth_dir)
        shutil.copy2(os.path.join(mask_dir, base_filename + '.npy'), dest_mask_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Subsample a dataset.')
    parser.add_argument('--src_dir', '-sd', required=True, type=str, help='Path to the source dataset directory')
    parser.add_argument('--dest_dir', '-dd', required=True, type=str, help='Path to the destination dataset directory')
    parser.add_argument('--downsample_ratio', '-dr', type=float, default=0.05, help='Downsample ratio for the dataset')
    args = parser.parse_args()

    subsample_dataset(args.src_dir, args.dest_dir, downsample_ratio=args.downsample_ratio)
