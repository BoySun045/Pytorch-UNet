import os
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm
import re


def find_folders(base_path, start_id, end_id):
    # This pattern matches "Actmap" followed by any characters, ending with a number of exactly five digits
    pattern = re.compile(r'Actmap.*(\d{5})$')
    
    # make start_id and end_id into int 
    start_id = int(start_id)
    end_id = int(end_id)

    # List to hold directories that match criteria
    matched_dirs = []
    
    # Iterate over all items in base_path
    for item in os.listdir(base_path):
        # Construct full path
        item_path = os.path.join(base_path, item)
        # Check if it's a directory
        if os.path.isdir(item_path):
            # Try to find a match with the pattern
            match = pattern.search(item)
            if match:
                # Extract the numerical part and convert to int
                folder_id = int(match.group(1))
                # Check if the id is within the specified range
                if start_id <= folder_id <= end_id:
                    matched_dirs.append(item_path)
    
    return matched_dirs

parser = argparse.ArgumentParser(description='flatten dataset')
parser.add_argument('--base_dir', '-bd', type=str, required=True)
parser.add_argument('--flat_data_dir', '-fd', type=str, required=True)
parser.add_argument('--start_id', '-s', type=int, required=True)
parser.add_argument('--end_id', '-e', type=int, required=True)
args = parser.parse_args()

# Define the base directory of your dataset
base_dir = Path(args.base_dir)

# Define the new, flat directory paths
flat_data_dir = Path(args.flat_data_dir)
flat_images_dir = Path(args.flat_data_dir + '/image')
flat_masks_dir = Path(args.flat_data_dir + '/mask')
flat_depths_dir = Path(args.flat_data_dir + '/depth')

# Create the directories if they don't exist
flat_data_dir.mkdir(parents=True, exist_ok=True)
flat_images_dir.mkdir(parents=True, exist_ok=True)
flat_masks_dir.mkdir(parents=True, exist_ok=True)
flat_depths_dir.mkdir(parents=True, exist_ok=True)

directory_path_list = find_folders(args.base_dir, args.start_id, args.end_id)
print(directory_path_list)

# Loop through each scene directory
for scene_dir in tqdm(directory_path_list):
    scene_dir = Path(scene_dir)
    if scene_dir.is_dir():
        scene_name = scene_dir.name
        # Process each file type (images, masks, depths)
        for file_type, output_dir in [('image', flat_images_dir), ('mask', flat_masks_dir), ('depth', flat_depths_dir)]:
            current_dir = scene_dir / file_type
            if current_dir.exists():
                for file in current_dir.iterdir():
                    if file.is_file():
                        # Construct the new filename with scene name as prefix
                        new_filename = f"{scene_name}_{file.name}"
                        new_filepath = output_dir / new_filename
                        # Copy the file to the new location
                        shutil.copy(file, new_filepath)

print("Dataset flattening complete.")
