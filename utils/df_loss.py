import torch.nn.functional as F
import torch 
import torch.nn as nn

def mse_loss(input, target):
    return F.mse_loss(input, target)

def mae_loss(input, target):
    return F.l1_loss(input, target)

def l1_loss_fn(input, target):
    return F.l1_loss(input, target)

def df_in_neighbor_loss(input, target, df_neighborhood=10):
    
    loss = 0
    # Retrieve the mask of valid pixels
    valid_mask = (target < df_neighborhood).float()
    print("valid_mask shape: ", valid_mask.shape)
    valid_norm = valid_mask.sum()
    print("valid_norm: ", valid_norm)
    valid_norm[valid_norm == 0] = 1


    df_loss = l1_loss_fn(input, target)
    print("df_loss shape: ", df_loss.shape)
    df_loss /= df_neighborhood

    df_loss = (df_loss * valid_mask).sum() / valid_norm

    loss = df_loss

    return loss


def normalize_df(df, df_neighborhood):
    return -torch.log(df / df_neighborhood + 1e-6)

def denormalize_df(df_norm, df_neighborhood):
    return torch.exp(-df_norm) * df_neighborhood
    
def df_normalized_loss_in_neighbor(input, target, df_neighborhood=10):
    # use the normalization loss from https://github.com/cvg/DeepLSD/blob/19eafc71d0c8de868f1b2b1f389efc265e07cda1/deeplsd/models/deeplsd.py#L81
    
    # first, compute the loss
    df_loss= l1_loss_fn(input, normalize_df(target, df_neighborhood))

    # and with supervision only on the lines neighborhood
        # Retrieve the mask of valid pixels
    valid_mask = (target < df_neighborhood).float()
    valid_norm = valid_mask.sum()

    df_loss = (df_loss * valid_mask).sum() / valid_norm + 1e-6

    return df_loss

def weighted_mse_loss(input, target, binary_mask_, increase_factor=1.0, avg_using_binary_mask=True):
    """
    Calculate MSE loss weighted by a binary mask, only considering errors where the mask is 1.
    
    Args:
    - input (torch.Tensor): The predictions from the model.
    - target (torch.Tensor): The ground truth values.
    - binary_mask (torch.Tensor): A binary mask where 1 indicates relevant pixels for loss calculation.
    - increase_factor (float): The factor by which to increase the loss at relevant pixels.
    
    Returns:
    - torch.Tensor: The calculated loss.
    """
    # # squeeze the input and target tensors 
    input = input.squeeze()
    target = target.squeeze()
    epsilon = 1e-5

    binary_mask = (target > epsilon).long()
    
    if input.dim() == 4:
        num_batches = input.shape[0]
    else:
        num_batches = 1    

    # Apply binary mask to increase weights only where binary_mask is 1
    weight_map = torch.ones_like(target)
    weight_map[binary_mask > 0] *= increase_factor
    # for the case where the mask is 0, we don't want to pass gradient, so we set the weight to 0
    weight_map[binary_mask < epsilon] = 0.0

    # check if the number of 0 values and 1 values in the binary mask sum up to the total number of pixels
    assert binary_mask.sum() + (binary_mask == 0).sum() == binary_mask.numel()
    # print("number of 1 values in the binary mask: ", binary_mask.sum())
    # print("number of 0 values in the binary mask: ", (binary_mask == 0).sum())
    
    # Calculate squared error
    squared_error = (input - target) ** 2

    # Apply weights
    weighted_squared_error = squared_error * weight_map

    # average the loss over batches
    weighted_squared_error = weighted_squared_error/num_batches

    # Compute the mean loss over all pixels
    if avg_using_binary_mask:
        loss = weighted_squared_error.sum() / binary_mask.sum()
    else:
        # use all pixels to calculate the average
        loss = weighted_squared_error.sum() / binary_mask.numel()

    # print("loss: ", loss)
    return loss
