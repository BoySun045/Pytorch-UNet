import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.regression_loss import weighted_mse_loss
from utils.dice_score import dice_coeff
from utils.df_loss import df_in_neighbor_loss
from utils.utils import downsample_torch_mask

@torch.inference_mode()
def evaluate(net, dataloader, device, amp, use_depth=False, 
             only_depth=False, head_mode="segmentation",
             reg_ds_factor=1.0):
    net.eval()
    num_val_batches = len(dataloader)

    loss_fn_rg = weighted_mse_loss
    dice_score = 0
    reg_loss = 0
    df_loss = 0
    loss_fn_df = df_in_neighbor_loss

    autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'

    with torch.autocast(autocast_device, enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            true_binary_mask = batch['binary_mask']

            if use_depth and not only_depth:
                depth = batch['depth']
                image = torch.cat((image, depth), dim=1)
                df = batch['df']
            elif use_depth and only_depth:
                image = batch['depth']
                df = batch['df']
            elif not use_depth and only_depth:
                raise ValueError('Cannot use only_depth without use_depth')

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.float32)
            true_binary_mask = true_binary_mask.to(device=device, dtype=torch.float32)
            true_df = df.to(device=device, dtype=torch.float32)

            ds_mask_true = downsample_torch_mask(mask_true, reg_ds_factor, ds_method='bilinear') if reg_ds_factor != 1.0 else mask_true
            ds_true_binary_mask = downsample_torch_mask(true_binary_mask, reg_ds_factor, ds_method='nearest') if reg_ds_factor != 1.0 else true_binary_mask
            ds_true_df = downsample_torch_mask(true_df, reg_ds_factor, ds_method='bilinear') if reg_ds_factor != 1.0 else true_df

            if head_mode == "segmentation":
                binary_pred = net(image)
                dice_score += dice_coeff((F.sigmoid(binary_pred) > 0.5).float().squeeze(1), true_binary_mask, reduce_batch_first=False)
            elif head_mode == "regression":
                mask_pred = net(image)
                reg_loss += loss_fn_rg(mask_pred, mask_true.float(), true_binary_mask.float(), increase_factor=1.0, avg_using_binary_mask=False)
            elif head_mode == "both":
                binary_pred, mask_pred = net(image)
                dice_score += dice_coeff((F.sigmoid(binary_pred) > 0.5).float().squeeze(1), true_binary_mask, reduce_batch_first=False)
                # reg_loss += loss_fn_rg(mask_pred, mask_true.float(), true_binary_mask.float(), increase_factor=1.0, avg_using_binary_mask=True)
                reg_loss += loss_fn_rg(mask_pred, ds_mask_true.float(), ds_true_binary_mask.float(), increase_factor=1.0, avg_using_binary_mask=False)

            elif head_mode == "df":
                df_pred = net(image)
                df_loss += loss_fn_df(df_pred.float().squeeze(1), ds_true_df, df_neighborhood=10)
                
    net.train()
    avg_dice_score = dice_score / num_val_batches if dice_score != 0 else 0
    avg_reg_loss = reg_loss / num_val_batches if reg_loss != 0 else 0
    avg_df_loss = df_loss / num_val_batches if df_loss != 0 else 0

    return avg_dice_score, avg_reg_loss, avg_df_loss