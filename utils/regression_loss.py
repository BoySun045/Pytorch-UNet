import torch.nn.functional as F
import torch 
import torch.nn as nn

def mse_loss(input, target):
    return F.mse_loss(input, target)

def mae_loss(input, target):
    return F.l1_loss(input, target)

# def weighted_mse_loss(input, target, increase_factor=2.0):

#     # Generate weight map
#     weight_map = torch.ones_like(target)
#     weight_map[target > 0.01] *= increase_factor

#     # Calculate squared error
#     squared_error = (input - target) ** 2

#     # Apply weights
#     weighted_squared_error = squared_error * weight_map

#     # Compute the mean loss
#     loss = weighted_squared_error.mean()

#     return loss

def weighted_mse_loss(input, target, binary_mask, increase_factor=1.0, avg_using_binary_mask=True):
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
    binary_mask = binary_mask.squeeze()
    
    if input.dim() == 4:
        num_batches = input.shape[0]
    else:
        num_batches = 1    

    # Apply binary mask to increase weights only where binary_mask is 1
    weight_map = torch.ones_like(target)
    weight_map[binary_mask > 0] *= increase_factor
    # for the case where the mask is 0, we don't want to pass gradient, so we set the weight to 0
    weight_map[binary_mask == 0] = 1.0

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

def weighted_huber_loss(input, target, binary_mask, delta=0.5, increase_factor=5.0, avg_using_binary_mask=True):
    """
    Calculate Huber loss weighted by a binary mask, only considering errors where the mask is 1.
    
    Args:
    - input (torch.Tensor): The predictions from the model.
    - target (torch.Tensor): The ground truth values.
    - binary_mask (torch.Tensor): A binary mask where 1 indicates relevant pixels for loss calculation.
    - delta (float): Threshold for Huber loss calculation.
    - increase_factor (float): The factor by which to increase the loss at relevant pixels.
    
    Returns:
    - torch.Tensor: The calculated loss.
    """
    # # squeeze the input and target tensors 
    input = input.squeeze()
    target = target.squeeze()
    binary_mask = binary_mask.squeeze()
    
    if input.dim() == 4:
        num_batches = input.shape[0]

    else:
        num_batches = 1    

    # Apply binary mask to increase weights only where binary_mask is 1
    weight_map = torch.ones_like(target)
    weight_map[binary_mask > 0] *= increase_factor
    # for the case where the mask is 0, we don't want to pass gradient, so we set the weight to 0
    weight_map[binary_mask == 0] = 1.0

    # check if the number of 0 values and 1 values in the binary mask sum up to the total number of pixels
    assert binary_mask.sum() + (binary_mask == 0).sum() == binary_mask.numel()
    
    # Instantiate Huber loss with specified delta
    criterion = nn.HuberLoss(delta=delta, reduction='none')

    # Calculate Huber loss
    huber_loss = criterion(input, target)

    # Apply weights
    weighted_huber_loss = huber_loss * weight_map

    # average the loss over batches
    weighted_huber_loss = weighted_huber_loss/num_batches

    # Compute the mean loss over all pixels
    if avg_using_binary_mask:
        loss = weighted_huber_loss.sum() / binary_mask.sum()
    else:
        # use all pixels to calculate the average
        loss = weighted_huber_loss.sum() / binary_mask.numel()

    return loss

