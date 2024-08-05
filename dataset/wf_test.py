import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os.path import splitext, isfile, join
from get_depth_discontinuity import *
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import median_filter
from hm3d_gt import load_image, log_transform_mask, min_max_scale, compute_df, compute_wf
import cv2
import argparse
import matplotlib.pyplot as plt
from scipy.ndimage import grey_dilation, median_filter
import os
from tqdm import tqdm
from pathlib import Path

def get_df(mask, depth, scale, df_neighbourhood=10):
    mask = np.array(mask)
    mask = np.array(Image.fromarray(mask).resize((int(mask.shape[1] * scale), int(mask.shape[0] * scale)), resample=Image.NEAREST))
    binary_mask = (mask > 0.0001).astype(np.float32)

    w = depth.shape[1]
    h = depth.shape[0]
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
    depth = cv2.resize(depth, (newW, newH), interpolation=cv2.INTER_CUBIC)
    depth = np.asarray(depth).astype(np.float32)
    
    assert mask.shape == depth.shape, f'Mask and depth should have the same size, but are {mask.shape} and {depth.shape}'

    df = compute_df(binary_mask, depth, df_neighbourhood)
    return df

def get_wf(mask, distance_field, scale, wf_neighbourhood=10.0):
    mask = np.array(mask)
    mask = np.array(Image.fromarray(mask).resize((int(mask.shape[1] * scale), int(mask.shape[0] * scale)), resample=Image.NEAREST))
    wf = compute_wf(mask, distance_field, wf_neighbourhood, extend=False)
    return wf

def preprocess(pil_img, scale, is_mask, is_depth=False, log_transform=True):

    if is_mask:
        # if it is mask, the input is directly a np array with weights value 
        # do a resize, normalization and return is enough
        mask = np.array(pil_img)
        mask = np.array(Image.fromarray(mask).resize((int(mask.shape[1] * scale), int(mask.shape[0] * scale)), resample=Image.NEAREST))

        mask_weight_global_max = 3000.0
        mask_weight_global_min = 1e-6
        
        if log_transform:
            mask = np.clip(mask, mask_weight_global_min, mask_weight_global_max)
        
        else:
            mask = np.clip(mask, mask_weight_global_min, mask_weight_global_max)
            mask = min_max_scale(mask, mask_weight_global_min, mask_weight_global_max)
            mask = np.clip(mask, 0, 1)

        binary_mask = (mask > 0.0001).astype(np.int64)
        return mask, binary_mask


    else:

        if is_depth:

            w = pil_img.shape[1]
            h = pil_img.shape[0]
            newW, newH = int(scale * w), int(scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
            pil_img = cv2.resize(pil_img, (newW, newH), interpolation=cv2.INTER_CUBIC)
            img = np.asarray(pil_img).astype(np.float32)
        
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            # depth should be with only one channel
            if not img.shape[0] == 1:
                # make it with only one channel but keep the same dimension (1, H, W)
                img = img[0:1, ...]

            # normalize depth 
            img_min = img.min()
            img_max = img.max()
            img = (img - img_min) / (img_max - img_min)

            return img

        if not is_depth:

            w, h = pil_img.size
            newW, newH = int(scale * w), int(scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
            pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
            img = np.asarray(pil_img)

            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img
    

def plot_result(image, depth, df, wd, mask, binary_mask):
    # plot the results as multiple using pyplot
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0, 0].imshow(np.transpose(image, (1, 2, 0)))
    axs[0, 0].set_title('Image')
    axs[0, 0].axis('on')

    axs[0, 1].imshow(depth[0], cmap='gray')
    axs[0, 1].set_title('Depth')
    axs[0, 1].axis('on')

    axs[0, 2].imshow(df, cmap='viridis')
    axs[0, 2].set_title('DF')
    axs[0, 2].axis('on')

    axs[1, 0].imshow(wd, cmap='viridis')
    axs[1, 0].set_title('WF')
    axs[1, 0].axis('on')

    axs[1, 1].imshow(mask, cmap='viridis')
    axs[1, 1].set_title('Mask')
    axs[1, 1].axis('on')

    axs[1, 2].imshow(binary_mask, cmap='gray')
    axs[1, 2].set_title('Binary Mask')
    axs[1, 2].axis('on')

    plt.show()



def densify_wf_batch(wf_batch, threshold=1e-6, median_kernel_size=3, max_iterations=10000, n_median_filters=8):
    # Apply median filter to the entire batch N times to remove noise
    wf_filtered_batch = np.array([median_filter(wf, size=median_kernel_size) for wf in wf_batch])
    # repeat the median filter N times
    while n_median_filters > 1:
        wf_filtered_batch = np.array([median_filter(wf, size=median_kernel_size) for wf in wf_filtered_batch])
        n_median_filters -= 1

    # Create a mask for values above the threshold for the entire batch
    mask_batch = wf_filtered_batch > threshold
    
    # If all values are below the threshold for all batch items, return the filtered batch
    if not mask_batch.any():
        return wf_filtered_batch
    
    # Use grey_dilation to propagate the nearest values above the threshold
    structuring_element = np.ones((3, 3))
    
    # Initial result with the filtered wf_batch
    densified_wf_batch = wf_filtered_batch.copy()
    
    # Perform a limited number of dilation steps for each batch item
    for _ in range(max_iterations):
        dilated_batch = np.array([grey_dilation(densified_wf, footprint=structuring_element) for densified_wf in densified_wf_batch])
        densified_wf_batch = np.where(mask_batch, densified_wf_batch, dilated_batch)
        mask_batch = densified_wf_batch > threshold
        if mask_batch.all():
            print("All values are above the threshold, stopping early")
            break
    
    return densified_wf_batch


def main():
    parser = argparse.ArgumentParser(description='Compute WF and DF from mask and depth')
    parser.add_argument('--dataset', type=str, help='Path to the dataset')
    parser.add_argument('--scale', type=float, default=1.0, help="Scale factor for the mask and depth")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for processing")
    args = parser.parse_args()

    img_dir = Path(join(args.dataset, 'image'))
    depth_dir = Path(join(args.dataset, 'depth'))
    mask_dir = Path(join(args.dataset, 'weighted_mask'))

    dense_mask_dir = join(args.dataset, 'dense_mask')
    dense_mask_vis_dir = join(args.dataset, 'dense_mask_vis')

    if not os.path.exists(dense_mask_dir):
        os.makedirs(dense_mask_dir)

    if not os.path.exists(dense_mask_vis_dir):
        os.makedirs(dense_mask_vis_dir)
                    
    # find all images in img_dir
    img_names = [f for f in os.listdir(img_dir) if isfile(join(img_dir, f))]

    batch_size = args.batch_size
    num_batches = (len(img_names) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches)):
        batch_img_names = img_names[i * batch_size: (i + 1) * batch_size]

        masks = []
        depths = []
        imgs = []
        dfs = []

        for img_name in batch_img_names:
            print(img_name)

            img_name = splitext(img_name)[0]
            mask_file = list(mask_dir.glob(img_name + '.*'))[0]
            depth_file = list(depth_dir.glob(img_name + '.*'))[0]

            mask = load_image(mask_file)
            depth = load_image(depth_file, load_depth=True)
            img = load_image(join(img_dir, img_name + '.jpg'))

            df = get_df(mask, depth, args.scale, 10)

            depth = preprocess(depth, args.scale, is_mask=False, is_depth=True)
            img = preprocess(img, args.scale, is_mask=False, is_depth=False)
            mask, _ = preprocess(mask, args.scale, is_mask=True)

            masks.append(mask)
            depths.append(depth)
            imgs.append(img)
            dfs.append(df)

        print("Batch loaded")
        masks = np.stack(masks)
        depths = np.stack(depths)
        imgs = np.stack(imgs)
        dfs = np.stack(dfs)

        wfs = np.array([get_wf(mask, df, 1.0, 5) for mask, df in zip(masks, dfs)])
        print(wfs.shape)
        wfs_dense = densify_wf_batch(wfs)

        for img_name, wd_dense in zip(batch_img_names, wfs_dense):
            img_name = splitext(img_name)[0]

            plt.imsave(join(dense_mask_vis_dir, img_name + '.png'), wd_dense, cmap='viridis')

            # Save the dense mask as a numpy array
            np.save(join(dense_mask_dir, img_name + '.npy'), wd_dense)
            print("wd_dense max min ", wd_dense.max(), wd_dense.min())

if __name__ == '__main__':
    main()
