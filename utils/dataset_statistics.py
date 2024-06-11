from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from concurrent.futures import ThreadPoolExecutor

dir_path = Path("/cluster/project/cvg/boysun/Actmap_v3")
dir_img = Path(dir_path / 'image/')
dir_mask = Path(dir_path / 'weighted_mask/')
dir_checkpoint = Path(dir_path / 'checkpoints/')
dir_depth = Path(dir_path / 'depth/')
dir_debug = Path("/cluster/scratch/boysun/debug/")

def get_npz_file_list_under_dir(dir_path):
    print("Getting npz files from dir: ", dir_path)
    npz_files = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.endswith('.npz')]
    return npz_files

def load_weighted_mask_from_npz(filename):
    return np.load(filename)['weights']

def process_file(npz_file):
    try:
        weighted_mask = load_weighted_mask_from_npz(dir_mask / npz_file)
        mask_max_value = weighted_mask.max()
        mask_mean_value = weighted_mask[weighted_mask > 0].mean()
        mask_median_value = np.median(weighted_mask[weighted_mask > 0])
        non_negative_pixel_ratio = np.count_nonzero(weighted_mask) / weighted_mask.size * 100
        return mask_max_value, mask_mean_value, mask_median_value, non_negative_pixel_ratio
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
    weighted_mask_npz_list = get_npz_file_list_under_dir(dir_mask)
    print("Load from dir: ", dir_mask)
    print("Number of npz files: ", len(weighted_mask_npz_list))

    with ThreadPoolExecutor(max_workers=8) as executor:
        # results = list(tqdm(executor.map(process_and_delete_file, weighted_mask_npz_list), total=len(weighted_mask_npz_list)))
        results = list(tqdm(executor.map(process_file, weighted_mask_npz_list), total=len(weighted_mask_npz_list)))


    mask_max_value_list = []
    mask_mean_value_list = []
    mask_median_value_list = []
    non_negative_pixel_ratio_list = []
    
    for mask_max_value, mask_mean_value, mask_median_value, non_negative_pixel_ratio in results:
        mask_max_value_list.append(mask_max_value)
        mask_mean_value_list.append(mask_mean_value)
        mask_median_value_list.append(mask_median_value)
        non_negative_pixel_ratio_list.append(non_negative_pixel_ratio)


    global_max_value = max(mask_max_value_list)

    # save result to a npy file
    np.save(dir_debug / 'mask_max_value_list.npy', np.array(mask_max_value_list))
    np.save(dir_debug / 'mask_mean_value_list.npy', np.array(mask_mean_value_list))
    np.save(dir_debug / 'non_negative_pixel_ratio_list.npy', np.array(non_negative_pixel_ratio_list))
    np.save(dir_debug / 'mask_median_value_list.npy', np.array(mask_median_value_list))

    # plot the distribution of max value, x axis is the max value, y axis is the number of masks, interval is 100
    # and save the result to the debug folder,
    plt.hist(mask_max_value_list, bins=range(0, int(global_max_value) + 20, 20))
    plt.xlabel('Max value of weighted mask')
    plt.ylabel('Number of masks')
    plt.title('Distribution of max value of weighted mask')
    plt.savefig(dir_debug / 'max_value_distribution.png')

    # do the same for the mean value
    # first, clear the current figure
    plt.clf()
    plt.hist(mask_mean_value_list, bins=range(0, int(global_max_value) + 20, 20))
    plt.xlabel('Mean value of weighted mask')
    plt.ylabel('Number of masks')
    plt.title('Distribution of mean value of weighted mask')
    plt.savefig(dir_debug / 'mean_value_distribution.png')

    #
    plt.clf()
    plt.hist(mask_median_value_list, bins=range(0, int(global_max_value) + 20, 20))
    plt.xlabel('Median value of weighted mask')
    plt.ylabel('Number of masks')
    plt.title('Distribution of median value of weighted mask')
    plt.savefig(dir_debug / 'median_value_distribution.png')


    # and then plot a 2D histogram of the max value and mean value, interval is the same as above, and save the result to the debug folder
    plt.clf()
    plt.hist2d(mask_max_value_list, mask_mean_value_list, bins=(range(0, int(global_max_value) + 100, 100), range(0, int(global_max_value) + 100, 100)))
    plt.xlabel('Max value of weighted mask')
    plt.ylabel('Mean value of weighted mask')
    plt.title('2D distribution of max value and mean value of weighted mask')
    plt.colorbar()
    plt.savefig(dir_debug / '2D_distribution.png')

    # plot the percentage of non-negative pixels in the mask, interval is 1 percent, and save the result to the debug folder
    plt.clf()
    plt.hist(non_negative_pixel_ratio_list, bins=np.linspace(0, 1, 101))
    plt.xlabel('Percentage of non-negative pixels in the mask')
    plt.ylabel('Number of masks')
    plt.title('Distribution of percentage of non-negative pixels in the mask')
    plt.savefig(dir_debug / 'non_negative_pixel_ratio_distribution.png')
