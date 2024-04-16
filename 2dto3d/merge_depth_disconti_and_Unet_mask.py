from get_depth_discontinuity import *
import cv2
import numpy as np
import os 
import sys
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    mask_normalized = mask_numpy / 255.0

    # Create a Gaussian kernel for smoothing
    # kernel_size should be positive and odd
    if kernel_size % 2 == 0:
        kernel_size += 1  # Make kernel size odd if it's not
    gaussian_kernel = cv2.getGaussianKernel(kernel_size, 0)
    kernel = np.outer(gaussian_kernel, gaussian_kernel)

    # Convolution to smooth the mask
    refined_mask = cv2.filter2D(mask_normalized, -1, kernel)

    # Convert to binary mask
    binary_mask = (refined_mask > 0).astype(np.uint8)

    return binary_mask

def mask_to_3d(mask_numpy, depth_numpy, gradient_direction_numpy=None):
    # Define the shape of the output array: [height, width, 2]
    depth_pairs = np.zeros((depth_numpy.shape[0], depth_numpy.shape[1], 2))

    if gradient_direction_numpy is None:
        _, gradient_direction = get_gradient_magitude_and_direction(depth_numpy)
    else:
        gradient_direction = gradient_direction_numpy
    
    # Iterate over each pixel in the mask
    for i in range(mask_numpy.shape[0]):
        for j in range(mask_numpy.shape[1]):
            if mask_numpy[i, j] == 1:  # Process only positive mask pixels
                direction = gradient_direction[i, j]
                di = int(np.round(np.sin(np.radians(direction))))
                dj = int(np.round(np.cos(np.radians(direction))))
                
                neighbor_i = i + di
                neighbor_j = j + dj

                # Check if the neighbor pixel is within the image bounds
                if 0 <= neighbor_i < depth_numpy.shape[0] and 0 <= neighbor_j < depth_numpy.shape[1]:
                    depth_pairs[i, j, 0] = depth_numpy[i, j]
                    depth_pairs[i, j, 1] = depth_numpy[neighbor_i, neighbor_j]
                else:
                    # If out of bounds, store zeros
                    depth_pairs[i, j, :] = 0
            else:
                # If not a positive mask pixel, store zeros
                depth_pairs[i, j, :] = 0

    return depth_pairs


def plot_depth_maps(depth_image, depth_pairs, downsample_factor=1):
    fig = plt.figure(figsize=(21, 6))  # Wider figure to accommodate three subplots

    # Downsample the depth data for visualization
    if downsample_factor > 1:
        depth_image = depth_image[::downsample_factor, ::downsample_factor]
        # depth_pairs = depth_pairs[::downsample_factor, ::downsample_factor]

    # Define camera position and direction
    camera_x = depth_image.shape[1] // 2
    camera_y = depth_image.shape[0] // 2

    # Plot the original depth image
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title('Original Depth Image')
    X, Y = np.meshgrid(np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0]))
    ax1.plot_surface(X, Y, depth_image, cmap='viridis')
    ax1.scatter([camera_x], [camera_y], [0], color='blue', s=100, label='Camera Viewpoint')
    ax1.quiver(camera_x, camera_y, 0, 0, 0, 1, length=50, color='blue', label='View Direction', arrow_length_ratio=0.1)
    ax1.legend()
    # set the axis limit for the first plot
    x_limit = depth_image.shape[1] * 1.1
    y_limit = depth_image.shape[0] * 1.1
    z_limit = np.max(depth_image) * 1.1
    ax1.set_xlim(0, x_limit)
    ax1.set_ylim(0, y_limit)
    ax1.set_zlim(0, z_limit)

    # Plot the refined depth pairs
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_xlim(0, x_limit)
    ax2.set_ylim(0, y_limit)
    ax2.set_zlim(0, z_limit)
    ax2.set_title('Depth from Depth Pairs')
    for i in range(depth_pairs.shape[0]):
        for j in range(depth_pairs.shape[1]):
            if depth_pairs[i, j, 0] != 0 and depth_pairs[i, j, 1] != 0:
                z1, z2 = depth_pairs[i, j]
                zs = np.linspace(z1, z2, num=int(abs(z2 - z1) / 0.10) + 1)
                xs = np.full(zs.shape, j)
                ys = np.full(zs.shape, i)
                ax2.plot(xs, ys, zs, 'red')
    ax2.scatter([camera_x], [camera_y], [0], color='blue', s=100, label='Camera Viewpoint')
    ax2.quiver(camera_x, camera_y, 0, 0, 0, 1, length=35, color='blue', label='View Direction', arrow_length_ratio=0.1)
    ax2.legend()

    # Combine both plots into a third subplot
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_xlim(0, x_limit)
    ax3.set_ylim(0, y_limit)
    ax3.set_zlim(0, z_limit)
    ax3.set_title('Combined Depth View')
    # Plot original depth image
    ax3.plot_surface(X, Y, depth_image, cmap='viridis', alpha=0.5)  # semi-transparent
    # Plot depth pairs
    for i in range(depth_pairs.shape[0]):
        for j in range(depth_pairs.shape[1]):
            if depth_pairs[i, j, 0] != 0 and depth_pairs[i, j, 1] != 0:
                z1, z2 = depth_pairs[i, j]
                zs = np.linspace(z1, z2, num=int(abs(z2 - z1) / 0.10) + 1)
                xs = np.full(zs.shape, j)
                ys = np.full(zs.shape, i)
                ax3.plot(xs, ys, zs, 'red')
    ax3.scatter([camera_x], [camera_y], [0], color='blue', s=100, label='Camera Viewpoint')
    ax3.quiver(camera_x, camera_y, 0, 0, 0, 1, length=50, color='blue', label='View Direction', arrow_length_ratio=0.1)
    ax3.legend()

    ax1.view_init(elev=-72, azim=-60, roll=0)
    ax2.view_init(elev=-72, azim=-60, roll=0)
    ax3.view_init(elev=-72, azim=-60, roll=0)

    plt.show()
def main():
    parser = argparse.ArgumentParser(description='Detect depth discontinuities and merge with Unet mask')
    parser.add_argument('--data_folder', type=str, help='Path to the folder containing the depth images and Unet masks')
    parser.add_argument('--scene_id', type=int, help='ID of the scene to process')
    parser.add_argument('--point_id', type=int, help='ID of the point to process')
    parser.add_argument('--vp_id', type=int, help='ID of the viewpoint to process')
    args = parser.parse_args()

    scene_id = "{:05d}".format(int(args.scene_id))
    point_id = args.point_id
    vp_id = args.vp_id
    scene_folder = os.path.join(args.data_folder, "Actmap_MH3D_{}".format(scene_id))
    depth_folder = os.path.join(scene_folder, "depth")
    mask_folder = os.path.join(scene_folder, "predict_mask_rgbd")
    depth_filename = "depth_{}_{}.png".format(point_id, vp_id)
    mask_filename = "img_{}_{}.jpg".format(point_id, vp_id)

    depth_path = os.path.join(depth_folder, depth_filename)
    mask_path = os.path.join(mask_folder, mask_filename)

    # Load a sample depth image (assuming it's loaded as a 2D numpy array)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    
    if depth_image is None:
        print("Error loading image")
        return

    grad_x, grad_y = compute_gradients(depth_image)
    magnitude, direction = gradient_magnitude_and_direction(grad_x, grad_y)
    
    # Threshold for detecting discontinuities
    threshold = 150  # This value might need tuning depending on the depth range
    discontinuity_mask = detect_discontinuities(magnitude, threshold)
    
    # load the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    refined_mask = refine_mask(mask, kernel_size=1)

    # the combined mask is the part that both the Unet mask and the discontinuity mask agree on
    combined_mask = np.logical_and(refined_mask, discontinuity_mask)

    # lift depth to 3d
    depth_pairs = mask_to_3d(combined_mask, depth_image, gradient_direction_numpy=direction)

    # Display results
    plt.figure(figsize=(15, 8))
    plt.subplot(231), plt.imshow(depth_image, cmap='gray'), plt.title('Original Depth Image')
    plt.subplot(234), plt.imshow(magnitude, cmap='gray'), plt.title('Depth Gradient Magnitude')
    plt.subplot(232), plt.imshow(mask, cmap='gray'), plt.title('Unet Mask')
    plt.subplot(233), plt.imshow(refined_mask, cmap='gray'), plt.title('Refined Unet Mask')
    # plt.subplot(233), plt.imshow(direction, cmap='gray'), plt.title('Gradient Direction')
    plt.subplot(236), plt.imshow(combined_mask, cmap='gray'), plt.title('Combined Mask')
    plt.subplot(235), plt.imshow(discontinuity_mask, cmap='gray'), plt.title('Depth Discontinuity Mask')
    plot_depth_maps(depth_image, depth_pairs)
    plt.show()

if __name__ == "__main__":
    main()
