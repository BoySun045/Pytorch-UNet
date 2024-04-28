import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.regression_loss import mse_loss, mae_loss, weighted_mse_loss
from utils.dice_score import dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, use_depth=False, only_depth=False):
    net.eval()
    num_val_batches = len(dataloader)

    loss_fn_rg = weighted_mse_loss
    dice_score = 0
    total_loss = 0
    rg_weight = 5.0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
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

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            true_binary_mask = true_binary_mask.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred, binary_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()

                reg_loss = loss_fn_rg(mask_pred, mask_true.float(), true_binary_mask.float())
                # compute the Dice score
                dice_score += dice_coeff(mask_pred.squeeze(1), mask_true, reduce_batch_first=False)
            else:
                print("for classification, number of classes should be 1")
                return

    net.train()
    return dice_score / max(num_val_batches, 1)
