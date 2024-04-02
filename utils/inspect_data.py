import os
import numpy as np
import cv2
import random
from pathlib import Path
import argparse

def visualize_random_datapoint(src_dir, dest_dir, scale=1.0):
    # Ensure the destination directory exists
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    
    # Get a list of all image filenames (assuming images are in JPG format)
    image_files = list(Path(src_dir, 'image').glob('*.jpg'))
    if not image_files:
        print("No image files found.")
        return
    
    # Randomly select a file
    selected_file = random.choice(image_files)
    base_name = selected_file.stem
    
    # Read image, depth, and mask
    image = cv2.imread(str(selected_file))
    depth = cv2.imread(str(Path(src_dir, 'depth', f'{base_name}.png')), cv2.IMREAD_UNCHANGED)
    mask = np.load(str(Path(src_dir, 'mask', f'{base_name}.npy')))

    # Resize image and depth with bicubic interpolation
    if scale != 1.0:
        new_dimensions = (int(scale * image.shape[1]), int(scale * image.shape[0]))
        image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_CUBIC)
        depth = cv2.resize(depth, new_dimensions, interpolation=cv2.INTER_CUBIC)
    
    # Convert binary mask to an image (0 -> black, 1 -> white) and resize with nearest neighbor interpolation
    mask_image = (mask * 255).astype(np.uint8)
    if scale != 1.0:
        mask_image = cv2.resize(mask_image, new_dimensions, interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, new_dimensions, interpolation=cv2.INTER_NEAREST)
    
    # Create a colored mask for overlay
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 1] = [0, 0, 255]  # Red
    
    # Overlay the colored mask on the image
    overlay_image = cv2.addWeighted(image, 1, colored_mask, 0.3, 0)
    
    # Save the visualizations
    cv2.imwrite(str(Path(dest_dir, f'{base_name}_image.jpg')), image)
    cv2.imwrite(str(Path(dest_dir, f'{base_name}_depth.png')), depth)
    cv2.imwrite(str(Path(dest_dir, f'{base_name}_mask.jpg')), mask_image)
    cv2.imwrite(str(Path(dest_dir, f'{base_name}_overlay.jpg')), overlay_image)
    
    print(f"Visualization for {base_name} saved to {dest_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize a random datapoint with mask overlay.')
    parser.add_argument('--src_dir', '-sd', required=True, type=str, help='Path to the source dataset directory')
    parser.add_argument('--dest_dir', '-dd', required=True, type=str, help='Path to the destination dataset directory')
    parser.add_argument('--scale', '-sc', required=False, type=float, default=1.0, help='Scale factor for resizing images')
    args = parser.parse_args()
    
    visualize_random_datapoint(args.src_dir, args.dest_dir, args.scale)
