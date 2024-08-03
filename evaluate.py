import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.regression_loss import weighted_mse_loss, masked_f1_loss, reverse_log_transform, log_transform
from utils.dice_score import dice_coeff
from utils.df_loss import df_in_neighbor_loss, l1_loss_fn, denormalize_df, mae_loss
from utils.utils import downsample_torch_mask

@torch.inference_mode()
def evaluate(net, dataloader, device, amp, use_depth=False, 
             use_mono_depth=False, head_mode="segmentation",
             reg_ds_factor=1.0):
    net.eval()
    num_val_batches = len(dataloader)

    # loss_fn_rg = masked_f1_loss
    loss_fn_rg = mae_loss
    dice_score = 0
    reg_loss = 0
    df_loss = 0
    loss_fn_df = mae_loss

    autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'

    with torch.autocast(autocast_device, enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            true_binary_mask = batch['binary_mask']
            depth = batch['depth'] if not use_mono_depth else batch['mono_depth']
            df = batch['df']

            if use_depth:    
                image = torch.cat((image, depth), dim=1)

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
                # reg_loss += loss_fn_rg(mask_pred, mask_true.float(), true_binary_mask.float(), increase_factor=1.0, avg_using_binary_mask=False)
                reg_loss += loss_fn_rg(mask_pred, mask_true.float())
            elif head_mode == "both":
                binary_pred, mask_pred = net(image)
                dice_score += dice_coeff((F.sigmoid(binary_pred) > 0.5).float().squeeze(1), true_binary_mask, reduce_batch_first=False)
                # reg_loss += loss_fn_rg(mask_pred, mask_true.float(), true_binary_mask.float(), increase_factor=1.0, avg_using_binary_mask=True)
                reg_loss += loss_fn_rg(mask_pred, ds_mask_true.float(), ds_true_binary_mask.float(), increase_factor=1.0, avg_using_binary_mask=False)

            elif head_mode == "df":
                df_pred = net(image)
                df_pred = denormalize_df(df_pred, df_neighborhood=10)
                df_loss += loss_fn_df(df_pred.float().squeeze(1), ds_true_df)
            
            elif head_mode == "df_wf":
                df_pred, mask_pred = net(image)
                print("df pred")
                df_pred = denormalize_df(df_pred, df_neighborhood=10)
                print("df_pred shape", df_pred.shape)
                print("true df shape", ds_true_df.shape)
                df_loss += loss_fn_df(df_pred.float().squeeze(1), ds_true_df)
                print("df loss, ", df_loss)
                # mask_true_log = log_transform(mask_true)
                reg_loss += loss_fn_rg(mask_pred.squeeze(), mask_true.float())

    net.train()
    avg_dice_score = dice_score / num_val_batches if dice_score != 0 else 0
    avg_reg_loss = reg_loss / num_val_batches if reg_loss != 0 else 0
    avg_df_loss = df_loss / num_val_batches if df_loss != 0 else 0

    return avg_dice_score, avg_reg_loss, avg_df_loss