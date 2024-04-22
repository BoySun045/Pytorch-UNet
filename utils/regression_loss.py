import torch.nn.functional as F
import torch 

def mse_loss(input, target):
    return F.mse_loss(input, target)

def mae_loss(input, target):
    return F.l1_loss(input, target)

def weighted_mse_loss(input, target, increase_factor=2.0):

    # Generate weight map
    weight_map = torch.ones_like(target)
    weight_map[target > 0.001] *= increase_factor

    # Calculate squared error
    squared_error = (input - target) ** 2

    # Apply weights
    weighted_squared_error = squared_error * weight_map

    # Compute the mean loss
    loss = weighted_squared_error.mean()

    return loss

