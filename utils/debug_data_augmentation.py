import os
from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image
from data_augmentation import CustomTransform
import argparse
import numpy as np
# Assuming the CustomTransform class and add_noise function are defined as provided above
from os.path import splitext, isfile, join

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext == '.npz':
        return Image.fromarray(np.load(filename)['depth'])
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)

def save_image(image, path):
    """Save a PIL image or a tensor to a file path."""
    if isinstance(image, torch.Tensor):
        image = to_pil_image(image)
    image.save(path, 'JPEG')

def save_mask_as_image(mask, path):
    # mask is a Image with binary, save it as an image for one value to be white and the other to be black
    mask = mask.convert('L')
    # print the unique values in the mask
    print(np.unique(mask))
    # it has only 0 and 1, save it as image for pixel value 0 to be black and 1 to be white
    mask_np = np.array(mask)
    mask_np[mask_np == 0] = 0
    mask_np[mask_np == 1] = 255
    # save use PIL
    mask = Image.fromarray(mask_np)
    mask.save(path, 'JPEG')

def main(data_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load image, depth, and mask
    image = load_image(data_root + "/image/img_2_1.jpg")
    depth = load_image(data_root + "/depth/depth_2_1.png")
    mask = load_image(data_root + "/mask/2_1.npy")

    # Initialize custom transformation
    custom_transform = CustomTransform()

    # Apply transformation
    transformed_image, transformed_depth, transformed_mask = custom_transform(image, depth, mask)

    # Save the transformed images
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_image(transformed_image, os.path.join(output_dir, 'transformed_image.jpg'))
    save_image(transformed_depth, os.path.join(output_dir, 'transformed_depth.jpg'))
    save_mask_as_image(transformed_mask, os.path.join(output_dir, 'transformed_mask.jpg'))

    # save the original images, they were all PIL images
    save_image(image, os.path.join(output_dir, 'original_image.jpg'))
    save_image(depth, os.path.join(output_dir, 'original_depth.png'))
    save_mask_as_image(mask, os.path.join(output_dir, 'original_mask.jpg'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()

    DATA_ROOT = args.data_root
    OUTPUT_DIR = args.output_dir

    main(DATA_ROOT, OUTPUT_DIR)