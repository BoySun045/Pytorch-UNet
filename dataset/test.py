from hm3d_gt import *

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from scipy.ndimage import distance_transform_edt
from os.path import splitext
import os

# Main function
def main(depth_image_path, binary_mask_path, output_path):
    mask_files = os.listdir(binary_mask_path)
    
    for mask_file in tqdm(mask_files):
        print("Processing file: ", mask_file)
        if mask_file.endswith('.npz'):
            mask_path = os.path.join(binary_mask_path, mask_file)
            weighted_mask = load_image(mask_path)
            
            mask = weighted_mask > 0 
            
            depth_file = os.path.splitext(mask_file)[0] + '.png'
            depth_path = os.path.join(depth_image_path, depth_file)
            if not os.path.exists(depth_path):
                print(f"Missing depth file for mask: {mask_file}")
                continue
            else:
                # depth_image = load_image(depth_path)
                depth_image = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
                if isinstance(depth_image, Image.Image):
                    depth_image = np.array(depth_image)
                if mask.shape != depth_image.shape:
                    print(f"Invalid mask dimensions: {mask_file}")
                    print(f"Expected: {depth_image.shape}, Actual: {mask.shape}")
                    continue
                
                fl_mask, discontinuity_mask, refine_mask = get_frontier_line_mask(mask, depth_image)
                distance_field = calculate_euclidean_distance_field(fl_mask)
                distance_field[distance_field > 10] = 10

                # Create a subplot
                fig, axs = plt.subplots(2, 3, figsize=(15, 15))

                axs[0, 0].imshow(mask, cmap='gray')
                axs[0, 0].set_title('Binary Mask')

                axs[0, 1].imshow(depth_image, cmap='gray')
                axs[0, 1].set_title('Depth Image')

                axs[1, 0].imshow(fl_mask, cmap='gray')
                axs[1, 0].set_title('Frontier Line Mask')

                im = axs[1, 1].imshow(distance_field, cmap='viridis', interpolation='nearest')
                fig.colorbar(im, ax=axs[1, 1])
                axs[1, 1].set_title('Euclidean Distance Field Heatmap')

                axs[1, 2].imshow(discontinuity_mask, cmap='gray')
                axs[1, 2].set_title('Discontinuity Mask')

                axs[0, 2].imshow(refine_mask, cmap='gray')
                axs[0, 2].set_title('Refined Mask')
                

                # Save the subplot
                output_file = os.path.splitext(mask_file)[0] + '_comparison.png'
                plt.savefig(os.path.join(output_path, output_file))



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate Euclidean Distance Field Heatmap")
    parser.add_argument('--dataset_dir', '-dr', type=str, help="Path to the dataset directory")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    depth_dir = Path(dataset_dir / 'depth/')
    mask_dir = Path(dataset_dir / 'weighted_mask/')

    output_dir = Path(dataset_dir / 'df/')
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    args = parser.parse_args()
    main(depth_dir, mask_dir, output_dir)
