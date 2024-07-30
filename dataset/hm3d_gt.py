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
from .get_depth_discontinuity import *
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import median_filter, grey_dilation, binary_dilation
import cv2
from scipy.spatial import cKDTree


def load_image(filename, load_depth=False):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    if ext == '.npz':
        weights = np.load(filename)['weights']
        return weights
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    elif ext in ['.png', '.jpg'] and load_depth:
        # print("Loading depth image: ", filename)
        # print("filename type: ", type(filename) )
        #convert filename to string
        filename = str(filename)
        return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    elif ext in ['.png', '.jpg'] and not load_depth:
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



def refine_mask(mask_numpy, kernel_size=5, erosion_iterations=1):
    """
    Refine a mask image by smoothing and shrinking.

    Args:
    mask_numpy (numpy array): The input mask image as a numpy array.
    kernel_size (int): Size of the Gaussian kernel for smoothing.
    erosion_iterations (int): Number of iterations for thfrom scipy.ndimage import median_filtere erosion operation.

    Returns:
    numpy array: The refined binary mask as a numpy array.
    """
    # Normalize the image to 0-1
    mask_numpy = mask_numpy.astype(np.float32)
    mask_max = mask_numpy.max()
    mask_min = mask_numpy.min()
    mask_normalized = min_max_scale(mask_numpy, mask_min, mask_max)

    # Create a Gaussian kernel for smoothing
    if kernel_size % 2 == 0:
        kernel_size += 1  # Make kernel size odd if it's not
    gaussian_kernel = cv2.getGaussianKernel(kernel_size, 0)
    kernel = np.outer(gaussian_kernel, gaussian_kernel)

    # Convolution to smooth the mask
    smoothed_mask = cv2.filter2D(mask_normalized, -1, kernel)

    # Convert to binary mask
    binary_mask = (smoothed_mask > 0.0).astype(np.uint8)

    # Create a larger kernel for more aggrfrom scipy.spatial import cKDTreeessive erosion
    if erosion_iterations > 0:
        larger_erosion_kernel = np.ones((3, 3), np.uint8)

        # Apply erosion with the larger kernel
        refined_mask = cv2.erode(binary_mask, larger_erosion_kernel, iterations=erosion_iterations)
    else:
        refined_mask = binary_mask

    return refined_mask

def get_frontier_line_mask(binary_mask, depth_image):
    
    if depth_image is None:
        print("Error loading image")
        return

    assert binary_mask.shape == depth_image.shape, "Mask and depth image must have the same shape"

    grad_x, grad_y = compute_gradients(depth_image)
    magnitude, direction = gradient_magnitude_and_direction(grad_x, grad_y)
    from scipy.ndimage import median_filter
    # Threshold for detecting discontinuities
    threshold_max = 1500  # This value might need tuning depending on the depth range
    threshold_min = 50
    discontinuity_mask = detect_discontinuities(magnitude, threshold_max, threshold_min)

    refined_mask = refine_mask(binary_mask, kernel_size=5, erosion_iterations=0)

    # the combined mask is the part that both the Unet mask and the discontinuity mask agree on
    combined_mask = np.logical_and(refined_mask, discontinuity_mask)

    return combined_mask, discontinuity_mask, refined_mask

def calculate_euclidean_distance_field(binary_grid):
    # Use the distance_transform_edt function to calculate the Euclidean distance field
    distance_field = distance_transform_edt(binary_grid == 0)
    return distance_field

def compute_df(mask, depth, line_neighborhood=10):
    fl_mask, _, _ = get_frontier_line_mask(mask, depth)
    distance_field = calculate_euclidean_distance_field(fl_mask)
    distance_field[distance_field > line_neighborhood] = line_neighborhood + 1e-3  # Clip the distance field
    return distance_field



# def extend_weight_mask(weight_mask, kernel_size=7):
#     # Pad the weight_mask to handle the borders
#     pad_size = kernel_size // 2
#     padded_mask = np.pad(weight_mask, pad_size, mode='constant', constant_values=0)
    
#     extended_mask = np.zeros_like(weight_mask)

#     # Iterate over each pixel in the weight_mask
#     for y in range(weight_mask.shape[0]):
#         for x in range(weight_mask.shape[1]):
#             # Extract the kernel around the current pixel
#             kernel = padded_mask[y:y + kernel_size, x:x + kernel_size]
            
#             # Get the non-zero values in the kernel
#             non_zero_values = kernel[kernel > 0]
            
#             if non_zero_values.size > 0:
#                 # Compute the median of the non-zero values
#                 median_value = np.median(non_zero_values)
#                 extended_mask[y, x] = median_value
#             else:
#                 # If no non-zero values, keep the original pixel value
#                 extended_mask[y, x] = weight_mask[y, x]
    
#     return extended_mask


def extend_weight_mask(weight_mask, kernel_size=9):
    # Create a structuring element (a larger one for more aggressive growth)
    structuring_element = np.ones((kernel_size, kernel_size))
    
    # Apply grey dilation to grow the regions aggressively
    extended_mask = grey_dilation(weight_mask, footprint=structuring_element)
    
    return extended_mask

def compute_wf(weight_mask, distance_field, line_neighborhood=10):
    # for weight field, it takes the value from weigh_mask, if it's coresponing distance field value is less than line_neighborhood
    
    weight_mask = extend_weight_mask(weight_mask)
    weight_mask[distance_field > line_neighborhood] = 0
    return weight_mask
