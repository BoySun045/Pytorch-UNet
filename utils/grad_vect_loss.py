import torch

def weighted_cosine_similarity_loss(input, target, binary_mask, increase_factor=1.0):
    """
    Calculate cosine similarity loss weighted by a binary mask, only considering errors where the mask is 1.
    
    Args:
    - input (torch.Tensor): The predictions from the model, normalized normals.
    - target (torch.Tensor): The ground truth values, normalized normals.
    - binary_mask (torch.Tensor): A binary mask where 1 indicates relevant pixels for loss calculation.
    - increase_factor (float): The factor by which to increase the loss at relevant pixels.
    
    Returns:
    - torch.Tensor: The calculated loss.
    """
    
    # Ensure the inputs are normalized
    input_norm = torch.nn.functional.normalize(input, p=2, dim=1)
    target_norm = torch.nn.functional.normalize(target, p=2, dim=1)

    # Calculate cosine similarity
    cos_similarity = (input_norm * target_norm).sum(dim=1)  # Dot product along channel dimension

    # Calculate cosine distance
    cos_distance = 1.0 - cos_similarity

    # Apply binary mask to focus only on the relevant parts
    masked_cos_distance = cos_distance * binary_mask

    # Apply the weighting
    weight_map = torch.ones_like(masked_cos_distance)
    weight_map[binary_mask > 0] *= increase_factor
    weight_map[binary_mask == 0] = 0  # Ensure no gradient flow where the mask is zero
    
    # Apply weights to the masked cosine distance
    weighted_cos_distance = masked_cos_distance * weight_map

    # Compute the mean loss over all pixels considered by the mask
    loss = weighted_cos_distance.sum() / binary_mask.sum()

    return loss
