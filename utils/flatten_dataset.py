import os
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm
import re

def find_folders(base_path, start_id, end_id):
    pattern = re.compile(r'Actmap.*(\d{5})$')
    start_id = int(start_id)
    end_id = int(end_id)
    matched_dirs = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            match = pattern.search(item)
            if match:
                folder_id = int(match.group(1))
                if start_id <= folder_id <= end_id:
                    matched_dirs.append(item_path)
    return matched_dirs

def extract_ids(directory, pattern):
    id_set = set()
    for file in directory.iterdir():
        if file.is_file():
            match = re.search(pattern, file.stem)  # Use .stem to ignore the file extension
            if match:
                id_set.add(match.group(0))
    return id_set

parser = argparse.ArgumentParser(description='Flatten dataset with ID consistency check')
parser.add_argument('--base_dir', '-bd', type=str, required=True)
parser.add_argument('--flat_data_dir', '-fd', type=str, required=True)
parser.add_argument('--start_id', '-s', type=int, required=True)
parser.add_argument('--end_id', '-e', type=int, required=True)
args = parser.parse_args()

base_dir = Path(args.base_dir)
flat_data_dir = Path(args.flat_data_dir)
flat_images_dir = Path(f'{args.flat_data_dir}/image')
flat_masks_dir = Path(f'{args.flat_data_dir}/mask')
flat_depths_dir = Path(f'{args.flat_data_dir}/depth')

flat_data_dir.mkdir(parents=True, exist_ok=True)
flat_images_dir.mkdir(parents=True, exist_ok=True)
flat_masks_dir.mkdir(parents=True, exist_ok=True)
flat_depths_dir.mkdir(parents=True, exist_ok=True)

directory_path_list = find_folders(args.base_dir, args.start_id, args.end_id)
print(directory_path_list)

pattern = r"(\d+_\d+)"  # Adjusted pattern to match the numerical ID part only

for scene_dir in tqdm(directory_path_list):
    scene_dir = Path(scene_dir)
    scene_name = scene_dir.name
    print(f"Processing scene: {scene_name}")

    image_ids = extract_ids(scene_dir / 'image', pattern)
    mask_ids = extract_ids(scene_dir / 'mask', pattern)
    depth_ids = extract_ids(scene_dir / 'depth', pattern)

    # Find common IDs across image, depth, and mask
    common_ids = image_ids.intersection(mask_ids).intersection(depth_ids)

    # Process each file type (images, masks, depths)
    for file_type, output_dir in [('image', flat_images_dir), ('mask', flat_masks_dir), ('depth', flat_depths_dir)]:
        current_dir = scene_dir / file_type
        if current_dir.exists():
            for file in current_dir.iterdir():
                if file.is_file():
                    match = re.search(pattern, file.stem)  # Again, use .stem to focus on the ID
                    if match and match.group(0) in common_ids:
                        file_suffix = match.group(0) + file.suffix  # Now add back the file extension for the new filename
                        new_filename = f"{scene_name}_{file_suffix}"
                        new_filepath = output_dir / new_filename
                        shutil.copy(file, new_filepath)

print("Dataset flattening complete with ID consistency check.")
