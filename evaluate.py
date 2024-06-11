import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.regression_loss import weighted_mse_loss
from utils.dice_score import dice_coeff

@torch.inference_mode()
def evaluate(net, dataloader, device, amp, use_depth=False, only_depth=False, head_mode="segmentation"):
    net.eval()
    num_val_batches = len(dataloader)

    loss_fn_rg = weighted_mse_loss
    dice_score = 0
    reg_loss = 0

    autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'

    with torch.autocast(autocast_device, enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            true_binary_mask = batch['binary_mask']

            if use_depth and not only_depth:
                depth = batch['depth']
                image = torch.cat((image, depth), dim=1)
            elif use_depth and only_depth:
                image = batch['depth']
            elif not use_depth and only_depth:
                raise ValueError('Cannot use only_depth without use_depth')

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.float32)
            true_binary_mask = true_binary_mask.to(device=device, dtype=torch.long)

            if head_mode == "segmentation":
                binary_pred = net(image)
                dice_score += dice_coeff((F.sigmoid(binary_pred) > 0.5).float().squeeze(1), true_binary_mask, reduce_batch_first=False)
            elif head_mode == "regression":
                mask_pred = net(image)
                reg_loss += loss_fn_rg(mask_pred, mask_true.float(), true_binary_mask.float(), increase_factor=1.0, avg_using_binary_mask=False)
            elif head_mode == "both":
                binary_pred, mask_pred = net(image)
                dice_score += dice_coeff((F.sigmoid(binary_pred) > 0.5).float().squeeze(1), true_binary_mask, reduce_batch_first=False)
                reg_loss += loss_fn_rg(mask_pred, mask_true.float(), true_binary_mask.float(), increase_factor=1.0, avg_using_binary_mask=True)

    net.train()
    avg_dice_score = dice_score / num_val_batches if dice_score != 0 else 0
    avg_reg_loss = reg_loss / num_val_batches if reg_loss != 0 else 0
    return avg_dice_score, avg_reg_loss