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
from scipy.ndimage import grey_dilation, median_filter, maximum_filter
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

    df, discontinuity_mask, refined_mask = compute_df(binary_mask, depth, df_neighbourhood)
    return df, discontinuity_mask, refined_mask


def plot_result(image, depth, df, discontinuity_mask, refined_mask, line_map, wf, mask):
    # plot the results as multiple using pyplot
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    # axs[0, 0].imshow(np.transpose(image, (1, 2, 0)))
    axs[0, 0].imshow(image)
    axs[0, 0].set_title('Image')
    axs[0, 0].axis('on')

    axs[0, 1].imshow(depth, cmap='gray')
    axs[0, 1].set_title('Depth')
    axs[0, 1].axis('on')

    axs[0, 2].imshow(df, cmap='viridis')
    axs[0, 2].set_title('DF')
    axs[0, 2].axis('on')

    axs[1, 0].imshow(discontinuity_mask, cmap='gray')
    axs[1, 0].set_title('Discontinuity Mask')
    axs[1, 0].axis('on')

    axs[1, 1].imshow(refined_mask, cmap='gray')
    axs[1, 1].set_title('Refined Mask')
    axs[1, 1].axis('on')

    axs[1, 2].imshow(line_map, cmap='gray')
    axs[1, 2].set_title('line_map')
    axs[1, 2].axis('on')

    # use hot colormap for wf
    # and show the colorbar
    axs[2, 0].imshow(wf, cmap='hot')
    axs[2, 0].set_title('WF')
    axs[2, 0].axis('on')
    plt.colorbar(axs[2, 0].imshow(wf, cmap='hot'), ax=axs[2, 0])

    axs[2, 1].imshow(mask, cmap='hot')
    axs[2, 1].set_title('Mask')
    axs[2, 1].axis('on')
    plt.colorbar(axs[2, 1].imshow(mask, cmap='hot'), ax=axs[2, 1])

    plt.savefig("/home/boysun/Pytorch-Unet-latest/dataset/result.png")
    plt.show()


def get_wf(df, mask, scale = 1.0, df_neighbourhood=10):


    # resize df, mask using input scale
    w = df.shape[1]
    h = df.shape[0]
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
    df = cv2.resize(df, (newW, newH), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (newW, newH), interpolation=cv2.INTER_NEAREST)


    line_map = np.zeros_like(df)
    line_map[df < df_neighbourhood] = 1
    wf = np.zeros_like(df)
    wf[line_map == 1] = mask[line_map == 1]

    print("wf max: ", np.max(wf))
    print("wf min: ", np.min(wf))
    print("0-0.2: ", np.sum((wf >= 0) & (wf < 0.2)))
    print("0.2-0.4: ", np.sum((wf >= 0.2) & (wf < 0.4)))
    print("0.4-0.6: ", np.sum((wf >= 0.4) & (wf < 0.6)))
    print("0.6-0.8: ", np.sum((wf >= 0.6) & (wf < 0.8)))
    print("0.8-1: ", np.sum((wf >= 0.8) & (wf <= 1)))  

    wf_filtered = wf
    # specify a filter for wf, take the max value in the neighborhood

    kernel_size = 3
    # wf_filtered = median_filter(wf, size=3)
    wf_filtered = maximum_filter(wf_filtered, size=kernel_size)
    # wf_filtered = median_filter(wf, size=3)
    wf_filtered = maximum_filter(wf_filtered, size=kernel_size)


    print("wf_filtered max: ", np.max(wf_filtered))
    print("wf_filtered min: ", np.min(wf_filtered))

    # print the interval [0-0.2], [0.2-0.4], [0.4-0.6], [0.6-0.8], [0.8-1]
    print("0-0.2: ", np.sum((wf_filtered >= 0) & (wf_filtered < 0.2)))
    print("0.2-0.4: ", np.sum((wf_filtered >= 0.2) & (wf_filtered < 0.4)))
    print("0.4-0.6: ", np.sum((wf_filtered >= 0.4) & (wf_filtered < 0.6)))
    print("0.6-0.8: ", np.sum((wf_filtered >= 0.6) & (wf_filtered < 0.8)))
    print("0.8-1: ", np.sum((wf_filtered >= 0.8) & (wf_filtered <= 1)))  

    return wf_filtered 

def main():
    parser = argparse.ArgumentParser(description='Compute df from mask and depth')
    parser.add_argument('--mask', type=str, help='Path to the mask image')
    parser.add_argument('--depth', type=str, help='Path to the depth image')
    parser.add_argument('--output', type=str, help='Path to the output df image')
    parser.add_argument('--image', type=str, help='Path to the image')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale factor to resize the mask and depth images')

    args = parser.parse_args()

    img = load_image(args.image)
    mask = load_image(args.mask)
    depth = load_image(args.depth, load_depth=True)

    bin_mask = (mask > 0.0001).astype(np.float32)
    df, discontinuity_mask, refined_mask = get_df(mask, depth, args.scale, 10)

    line_map = np.zeros_like(df)
    line_map[df < 0.5] = 1

    wf = get_wf(df, mask, 1.0, 5)

    plot_result(img, depth, df, discontinuity_mask, refined_mask, line_map, wf, mask)


if __name__ == '__main__':
    main()