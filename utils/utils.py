import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()

def downsample_torch_mask(mask, factor, ds_method='bilinear'):
    """
    Downsamples a batch of torch image by a given factor using bilinear interpolation or nearest neighbor.
    for each image in the batch, do the downsampling operation, either bilinear or nearest neighbor.
    
    Args
    - image (torch.Tensor): The input image tensor. dimensions: (batch_size, height, width), dtype: float32.
    - factor (float): The downsample factor.
    
    Returns:
    - torch.Tensor: The downsampled image tensor. dimensions: (batch_size, height//factor, width//factor), dtype: float32.
    """

    if factor == 1.0:
        return mask

    # check if it is 4D input, if not, add a dimension in channel axis as the input to F.interpolate should be 4D
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)


    if ds_method == 'bilinear':
        mode = 'bilinear'
    elif ds_method == 'nearest':
        mode = 'nearest'
    else:
        raise ValueError(f"Downsampling method {ds_method} not recognized. Use 'bilinear' or 'nearest'.")

    return F.interpolate(mask, scale_factor=factor, mode=mode).squeeze(1)

    