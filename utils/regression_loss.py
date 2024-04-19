import torch.nn.functional as F
import torch 

def mse_loss(input, target):
    return F.mse_loss(input, target)

def mae_loss(input, target):
    return F.l1_loss(input, target)

def weighted_mse_loss(input, target, increase_factor=30):
    print("input is ", input)
    print("target is ", target)
    # Generate weight map
    weight_map = torch.ones_like(target)
    weight_map[target > 0.001] *= increase_factor

    # Calculate squared error
    squared_error = (input - target) ** 2

    # Apply weights
    weighted_squared_error = squared_error * weight_map
    print("weighted squared error is ", weighted_squared_error)

    # Compute the mean loss
    loss = weighted_squared_error.mean()

    return loss

