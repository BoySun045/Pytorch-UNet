import numpy as np
import os
from pathlib import Path
import argparse

def normalize_and_save(input_dir, output_dir, global_max):
    """
    Process all .npz files in the input directory, normalize the 'weights' array,
    save as .npy file, and also create binary masks.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists

    # Iterate over each .npz file in the directory
    for npz_file in input_dir.glob('*.npz'):
        data = np.load(npz_file)['weights']  # Load the 'weights' array from .npz
        normalized_data = data / global_max  # Normalize the data

        # Save normalized data as .npy
        npy_filename = output_dir / (npz_file.stem + '.npy')
        np.save(npy_filename, normalized_data)

        # Create binary mask and save
        binary_mask = (normalized_data > 0.5).astype(np.float32)  # Create binary mask
        mask_filename = output_dir / (npz_file.stem + '_mask.npy')
        np.save(mask_filename, binary_mask)

        print(f"Processed {npz_file.name}: saved {npy_filename} and {mask_filename}")

def main():
    parser = argparse.ArgumentParser(description='Normalize .npz weights and convert to .npy and binary masks.')
    parser.add_argument('--input_dir', type=str, help='Directory containing .npz files')
    parser.add_argument('--output_dir', type=str, help='Directory to save .npy files and masks')
    parser.add_argument('--global_max', type=float, help='Global maximum value for normalization')
    args = parser.parse_args()

    normalize_and_save(args.input_dir, args.output_dir, args.global_max)

if __name__ == "__main__":
    main()
