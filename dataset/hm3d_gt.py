import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from get_depth_discontinuity import *
from scipy.ndimage import distance_transform_edt

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    if ext == '.npz':
        weights = np.load(filename)['weights']
        return weights
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)
    
# Log transformation
def log_transform_mask(y):
    return np.log1p(y)  # log1p(x) = log(1 + x)

# Reverse log transformation
def reverse_log_transform_mask(y):
    return np.expm1(y)  # expm1(x) = exp(x) - 1

def min_max_scale(y, min_val, max_val):
    return (y - min_val) / (max_val - min_val)

def reverse_min_max_scale(y, min_val, max_val):
    return y * (max_val - min_val) + min_val



def refine_mask(mask_numpy, kernel_size=5):
    """
    Refine a mask image by smoothing and filling holes.

    Args:
    image_path (str): Path to the mask image in JPEG format.
    kernel_size (int): Size of the Gaussian kernel for smoothing.

    Returns:
    numpy array: The refined binary mask as a numpy array.
    """
    # Normalize the image to 0-1

    mask_numpy = mask_numpy.astype(np.float32)
    mask_max = mask_numpy.max()
    mask_min = mask_numpy.min()
    mask_normalized = min_max_scale(mask_numpy, mask_min, mask_max)

    mask_normalized = mask_normalized.astype(np.float32)

    # Create a Gaussian kernel for smoothing
    # kernel_size should be positive and odd
    if kernel_size % 2 == 0:
        kernel_size += 1  # Make kernel size odd if it's not
    gaussian_kernel = cv2.getGaussianKernel(kernel_size, 0)
    kernel = np.outer(gaussian_kernel, gaussian_kernel)

    # Convolution to smooth the mask
    refined_mask = cv2.filter2D(mask_normalized, -1, kernel)

    # Convert to binary mask
    binary_mask = (refined_mask > 0.8).astype(np.uint8)

    return binary_mask

def get_frontier_line_mask(binary_mask, depth_image):
    
    if depth_image is None:
        print("Error loading image")
        return

    assert binary_mask.shape == depth_image.shape, "Mask and depth image must have the same shape"

    grad_x, grad_y = compute_gradients(depth_image)
    magnitude, direction = gradient_magnitude_and_direction(grad_x, grad_y)
    
    # Threshold for detecting discontinuities
    threshold = 200  # This value might need tuning depending on the depth range
    discontinuity_mask = detect_discontinuities(magnitude, threshold)

    refined_mask = refine_mask(binary_mask, kernel_size=5)

    # the combined mask is the part that both the Unet mask and the discontinuity mask agree on
    combined_mask = np.logical_and(refined_mask, discontinuity_mask)

    return combined_mask, discontinuity_mask, refined_mask

def calculate_euclidean_distance_field(binary_grid):
    # Use the distance_transform_edt function to calculate the Euclidean distance field
    distance_field = distance_transform_edt(binary_grid == 0)
    return distance_field