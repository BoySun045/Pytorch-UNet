import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

def validate_dataset(image_folder, mask_folder, image_width, image_height):
    # List to store paths of wrong images and masks
    wrong_files = []

    # Get list of files in image folder
    image_files = os.listdir(image_folder)

    # Iterate over image files
    for image_file in tqdm(image_files):
        # Check if file is a jpg image
        if image_file.endswith('.jpg'):
            image_path = os.path.join(image_folder, image_file)

            # Open image and check dimensions
            img = Image.open(image_path)
            img_width, img_height = img.size
            if img_width != image_width or img_height != image_height:
                print(f"Invalid image dimensions: {image_file}")
                print(f"Expected: ({image_width}, {image_height}), Actual: ({img_width}, {img_height})")
                wrong_files.append(image_path)

            # Check if corresponding mask file exists
            mask_file = os.path.splitext(image_file)[0] + '.npz'
            mask_path = os.path.join(mask_folder, mask_file)
            if not os.path.exists(mask_path):
                print(f"Missing mask file for image: {image_file}")
                wrong_files.append(mask_path)
            else:
                # Load mask npz file and check dimensions
                mask_data = np.load(mask_path)
                mask_array = mask_data['weights']
                if mask_array.shape != (image_height, image_width):
                    print(f"Invalid mask dimensions: {mask_file}")
                    print(f"Expected: ({image_width}, {image_height}), Actual: {mask_array.shape}")
                    wrong_files.append(mask_path)

    # Delete wrong files
    for file_path in wrong_files:
        try:
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Error deleting file: {file_path}, {e}")

def main():
    parser = argparse.ArgumentParser(description='Validate dataset images and masks')
    parser.add_argument('--image_folder', '-if', type=str, help='Path to folder containing images')
    parser.add_argument('--mask_folder', '-mf', type=str, help='Path to folder containing masks')
    parser.add_argument('--image_width', '-width', type=int, default=720, help='Expected image width')
    parser.add_argument('--image_height', '-height', type=int, default=540, help='Expected image height')
    args = parser.parse_args()

    # Validate dataset
    validate_dataset(args.image_folder, args.mask_folder, args.image_width, args.image_height)

if __name__ == "__main__":
    main()
