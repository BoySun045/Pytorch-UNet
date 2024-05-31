from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

# dir_path = Path("/cluster/project/cvg/boysun/Actmap_v2")
dir_path = Path("/media/boysun/Extreme Pro/Actmap_v2_mini")
dir_img = Path(dir_path / 'image/')
dir_mask = Path(dir_path / 'weighted_mask/')
dir_checkpoint = Path(dir_path / 'checkpoints/')
dir_debug = Path(dir_path / 'debug/')
dir_depth = Path(dir_path / 'depth/')


def get_npz_file_list_under_dir(dir_path):
    npz_files = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.endswith('.npz')]
    return npz_files

def load_weighted_mask_from_npz(filename):
    return np.load(filename)['weights']


if __name__ == "__main__":
    
    weighted_mask_npz_list = get_npz_file_list_under_dir(dir_mask)
    print("load from dir: ", dir_mask)
    print("number of npz files: ", len(weighted_mask_npz_list))

    mask_max_value_list = []
    mask_mean_value_list = []

    for i, npz_file in tqdm(enumerate(weighted_mask_npz_list)):
        print(f"Processing {i+1}/{len(weighted_mask_npz_list)}")
        weighted_mask = load_weighted_mask_from_npz(dir_mask / npz_file)
        print("weighted_mask shape: ", weighted_mask.shape)
        mask_max_value_list.append(weighted_mask.max())
        mask_mean_value_list.append(weighted_mask.mean())
        global_max_value = max(mask_max_value_list)

    # plot the distribution of max value, x axis is the max value, y axis is the number of masks, interval is 100
    # and save the result to the debug folder,
    plt.hist(mask_max_value_list, bins=range(0, int(global_max_value)+100, 100))
    plt.xlabel('Max value of weighted mask')
    plt.ylabel('Number of masks')
    plt.title('Distribution of max value of weighted mask')
    plt.savefig(dir_debug / 'max_value_distribution.png')

    # do the same for the mean value
    # first, clear the current figure
    plt.clf()
    plt.hist(mask_mean_value_list, bins=range(0, int(global_max_value)+100, 100))
    plt.xlabel('Mean value of weighted mask')
    plt.ylabel('Number of masks')
    plt.title('Distribution of mean value of weighted mask')
    plt.savefig(dir_debug / 'mean_value_distribution.png')

    # and then plot a 2D histogram of the max value and mean value, interval is the same as above, and save the result to the debug folder
    plt.clf()
    plt.hist2d(mask_max_value_list, mask_mean_value_list, bins=(range(0, int(global_max_value)+100, 100), range(0, int(global_max_value)+100, 100)))
    plt.xlabel('Max value of weighted mask')
    plt.ylabel('Mean value of weighted mask')
    plt.title('2D distribution of max value and mean value of weighted mask')
    plt.colorbar()
    plt.savefig(dir_debug / '2D_distribution.png')





    

