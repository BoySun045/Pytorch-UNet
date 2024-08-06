from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import functools
import argparse

dir_path = Path("/cluster/project/cvg/boysun/Actmap_v3")
# dir_path = Path("/mnt/boysunSSD/Actmap_v2_mini")
dir_img = Path(dir_path / 'image/')
dir_mask = Path(dir_path / 'weighted_mask/')
dir_depth = Path(dir_path / 'depth/')
dir_debug = Path(dir_path / "debug")

if not dir_debug.exists():
    dir_debug.mkdir()

# Log transformation
def log_transform_mask(y):
    return np.log1p(y)  # log1p(x) = log(1 + x)

def create_label_mask_fast(array, num_bins, bin_edges=None):
    max = 3000
    array = np.clip(array, 0, max)  # Clip values to the range [0, 3000]
    log_max = np.log1p(max)
    array = log_transform_mask(array)
    if bin_edges==None:
        # bin_edges = np.arange(0, 3100, 100)
        # bin_edges = np.arange(0, 8, 0.25)  # create actually 33 bins ,0), (0, 0.25), (0.25, 0.5), ..., (7.75, 8)
        exp_bins = np.geomspace(1, 20, num_bins)[::-1]
        bin_edges = 8.5 - (exp_bins - exp_bins.min()) / (exp_bins.max() - exp_bins.min()) * 8.5
    # Use digitize to get the bin index for each element in the array
    label_mask = np.digitize(array, bin_edges) - 1  # Subtract 1 to make bins 0-indexed, since we do clip min to be 0, there will be no -1 bin
    return label_mask

def count_classes(label_mask, num_classes):
    # Initialize an array to hold the counts for each class
    class_counts = np.zeros(num_classes, dtype=int)
    
    # Use np.bincount to count the occurrences of each class in the label mask
    counts = np.bincount(label_mask.ravel(), minlength=num_classes)
    
    # Assign the counts to the class_counts array
    class_counts[:len(counts)] = counts
    
    return class_counts
    
def save_semantic_map(label_mask, npz_file):
    # Create an RGB image where each class is represented by a unique color
    height, width = label_mask.shape
    semantic_map = np.zeros((height, width, 3), dtype=np.uint8)

    # Define a colormap (hot colormap)
    colormap = plt.get_cmap('hot')  # 'hot' colormap

    for class_id in range(30):
        color = np.array(colormap(class_id / 29)[:3]) * 255  # Normalize class_id to range [0, 1]
        semantic_map[label_mask == class_id] = color

    # Save the image
    semantic_map_img = Image.fromarray(semantic_map)
    semantic_map_img.save(dir_debug / (splitext(npz_file)[0] + '.jpg'))


def get_npz_file_list_under_dir(dir_path):
    print("Getting npz files from dir: ", dir_path)
    npz_files = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.endswith('.npz')]
    return npz_files

def load_weighted_mask_from_npz(filename):
    return np.load(filename)['weights']

# def process_file(npz_file):
#     try:
#         weighted_mask = load_weighted_mask_from_npz(dir_mask / npz_file)
#         mask_max_value = weighted_mask.max()
#         mask_mean_value = weighted_mask[weighted_mask > 0].mean()
#         mask_median_value = np.median(weighted_mask[weighted_mask > 0])
#         non_negative_pixel_ratio = np.count_nonzero(weighted_mask) / weighted_mask.size * 100
#         return mask_max_value, mask_mean_value, mask_median_value, non_negative_pixel_ratio
#     except Exception as e:
#         print(f"Error processing file {npz_file}: {e}")
#         return None

def process_file(npz_file, num_bins=30):
    try:
        weighted_mask = load_weighted_mask_from_npz(dir_mask / npz_file)
        label_mask = create_label_mask_fast(weighted_mask ,num_bins )
        class_counts = count_classes(label_mask, num_bins)
        max_value = weighted_mask.max()
        # Save the semantic map as an RGB image
        # save_semantic_map(label_mask, npz_file)

        return class_counts, max_value
    
    except Exception as e:
        print(f"Error processing file {npz_file}: {e}")
        return None
    
def delet_file_with_prob(filename, prob):
    if np.random.rand() < prob:
        filename_no_ext = splitext(filename)[0]
        image_file = dir_img / (filename_no_ext + '.jpg')
        mask_file = dir_mask / (filename_no_ext + '.npz')
        depth_file = dir_depth / (filename_no_ext + '.png')
        try:
            if image_file.exists():
                image_file.unlink()
            if mask_file.exists():
                mask_file.unlink()
            if depth_file.exists():
                depth_file.unlink()
        except Exception as e:
            print(f"Error deleting files for {filename}: {e}")

def process_and_delete_file(npz_file):
    try:
        weighted_mask = load_weighted_mask_from_npz(dir_mask / npz_file)
        mask_max_value = weighted_mask.max()
        mask_mean_value = weighted_mask[weighted_mask > 0].mean()
        mask_median_value = np.median(weighted_mask[weighted_mask > 0])
        non_negative_pixel_ratio = np.count_nonzero(weighted_mask) / weighted_mask.size * 100
        
        if mask_median_value < 20:
            delet_file_with_prob(npz_file, 0.6)
        elif mask_median_value < 40:
            delet_file_with_prob(npz_file, 0.5)
        elif mask_median_value < 60:
            delet_file_with_prob(npz_file, 0.4)
        elif mask_median_value < 80:
            delet_file_with_prob(npz_file, 0.3)
        elif mask_median_value < 100:
            delet_file_with_prob(npz_file, 0.2)
        elif mask_median_value < 120:
            delet_file_with_prob(npz_file, 0.1)
        
        return mask_max_value, mask_mean_value, mask_median_value, non_negative_pixel_ratio
    except Exception as e:
        print(f"Error processing and deleting file {npz_file}: {e}")
        return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=' .')
    parser.add_argument('--file_suffix', '-fs',type=str, help='Path to the class_counts.npy file')

    suffix = parser.parse_args().file_suffix

    weighted_mask_npz_list = get_npz_file_list_under_dir(dir_mask)
    print("Load from dir: ", dir_mask)
    print("Number of npz files: ", len(weighted_mask_npz_list))

    num_bins = 30

    total_class_counts = np.zeros(num_bins, dtype=int)
    max_values = []

    # Create a partial function with num_bins as an argument
    process_file_partial = functools.partial(process_file, num_bins=num_bins)

    with ThreadPoolExecutor(max_workers=32) as executor:
        results = list(tqdm(executor.map(process_file_partial, weighted_mask_npz_list), total=len(weighted_mask_npz_list)))

    # Aggregate class counts
    for result in results:
        if result is not None:
            class_counts, max_value = result
            total_class_counts += class_counts
        if max_value is not None:
            max_values.append(max_value)

    # Print and save overall statistics
    print("\nOverall class counts:")
    for i, count in enumerate(total_class_counts):
        print(f"Class {i}: {count} pixels")

    with open(dir_debug / "class_counts_{}.txt".format(suffix), "w") as f:
        for i, count in enumerate(total_class_counts):
            f.write(f"Class {i}: {count} pixels\n")
    
    # Save class counts to a .npy file
    np.save(dir_debug / "class_counts_{}.npy".format(suffix), total_class_counts)
    
    # Save max values in both .txt and .npy formats
    with open(dir_debug / "max_values_{}.txt".format(suffix), "w") as f:
        for idx, max_val in enumerate(max_values):
            f.write(f"File {idx}: Max value = {max_val}\n")
    
    # Save max values to a .npy file
    np.save(dir_debug / "max_values_{}.npy".format(suffix), np.array(max_values))
    