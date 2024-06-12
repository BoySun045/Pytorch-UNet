import numpy as np
import cv2  # OpenCV for image processing
from matplotlib import pyplot as plt

def compute_gradients(depth_image):
    """
    Compute horizontal and vertical gradients of the depth image using the Sobel operator.

    Args:
    depth_image (numpy array): The depth image as a 2D numpy array.

    Returns:
    Tuple[numpy array, numpy array]: The horizontal and vertical gradients of the depth image.
    """
    if depth_image.dtype != np.float64:
        depth_image = depth_image.astype(np.float64)
    
    # Calculate gradients in x and y directions
    grad_x = cv2.Sobel(depth_image, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(depth_image, cv2.CV_64F, 0, 1, ksize=5)

    return grad_x, grad_y

def gradient_magnitude_and_direction(grad_x, grad_y):
    """
    Calculate the magnitude and direction of the gradient from horizontal and vertical gradients.

    Args:
    grad_x (numpy array): Gradient of the image in the x direction.
    grad_y (numpy array): Gradient of the image in the y direction.

    Returns:
    Tuple[numpy array, numpy array]: The magnitude and direction of the gradient.
    """
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x) * 180 / np.pi
    return magnitude, direction

def detect_discontinuities(magnitude, threshold_max, threshold_min=0):
    """
    Detect discontinuities by thresholding the gradient magnitude.

    Args:
    magnitude (numpy array): The magnitude of the gradient.
    threshold (float): Threshold value to detect edges.

    Returns:
    numpy array: A binary mask where discontinuities are marked.
    """
    discontinuity_mask = (magnitude > threshold_min) & (magnitude < threshold_max)

    return discontinuity_mask.astype(np.uint8)


def get_gradient_magitude_and_direction(depth_image):
    grad_x, grad_y = compute_gradients(depth_image)
    magnitude, direction = gradient_magnitude_and_direction(grad_x, grad_y)
    return magnitude, direction

def get_depth_discontinuity(depth_image, threshold):
    magnitude, _ = get_gradient_magitude_and_direction(depth_image)
    discontinuity_mask = detect_discontinuities(magnitude, threshold)
    return discontinuity_mask