import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.regression_loss import mse_loss, mae_loss, weighted_mse_loss
from utils.dice_score import dice_coeff

@torch.inference_mode()
def evaluate(net, dataloader, device, amp, use_depth=False, only_depth=False):
    net.eval()
    num_val_batches = len(dataloader)
    total_loss = 0
    
    # regresssion
    # loss_fn = weighted_mse_loss
    
    # binary classification
    loss_fn = dice_coeff
    dice_score = 0
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            if use_depth and not only_depth:
                depth = batch['depth']
                image = torch.cat((image, depth), dim=1)
            elif use_depth and only_depth:
                image = batch['depth']
            elif not use_depth and only_depth:
                raise ValueError('Cannot use only_depth without use_depth')

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.float32)

            # predict the mask
            mask_pred = net(image)
            mask_pred = mask_pred.squeeze(1)

            # calculate the loss
            # total_loss += mse_loss(mask_pred, mask_true).item()
            mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            # compute the Dice score
            dice_score += dice_coeff(mask_pred.squeeze(1), mask_true, reduce_batch_first=False)

    net.train()
    avg_loss = total_loss / max(num_val_batches, 1)
    
    return avg_loss
