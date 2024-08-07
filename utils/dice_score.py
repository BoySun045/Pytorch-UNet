import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def dice_coeff(input: Tensor, target: Tensor, valid_mask, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Ensure input, target, and valid_mask are of the same size
    
    if valid_mask is None:
        valid_mask = torch.ones_like(input)
    
    assert input.size() == target.size() == valid_mask.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    # Apply valid_mask
    input = input * valid_mask
    target = target * valid_mask

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def multiclass_dice_coeff(input: Tensor, target: Tensor, valid_mask: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Flatten the tensors for multiclass case
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), valid_mask.flatten(0, 1), reduce_batch_first, epsilon)

def dice_loss(input: Tensor, target: Tensor, valid_mask: Tensor, multiclass: bool = False):
    # Choose the correct function based on multiclass flag
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, valid_mask, reduce_batch_first=True)

def weighted_mask_cross_entropy_loss(ignore_idx: int = 0, weights = None, num_classes: int = 2):
    # Cross-entropy loss for segmentation masks
    # weights is a np.array of shape (num_classes,) with the weights for each class

    # igore_idx is the index of the class to ignore, update the weights to remove the ignored_idx'th class
    # weights = np.delete(weights, ignore_idx)
    
    if weights is None:
        norm_weights = np.ones(num_classes)
    else:  
        print(f"weights is {weights}")

        # take the number of classes into account
        weights[num_classes -1 ] = np.sum(weights[num_classes - 1:])
        weights = weights[:num_classes]
        print(f"weights after num_classes is {weights}")
        
        # normalize the weights using the inverse of the weights input
        weights = 1 / (weights + 1)
        print(f"weights after inverse is {weights}")
        norm_weights = weights / weights.sum()
        print(f"norm_weights is {norm_weights}")

    
    norm_weights = torch.tensor(norm_weights, dtype=torch.float32).cuda()
    return nn.CrossEntropyLoss(weight=norm_weights, ignore_index=ignore_idx)

    