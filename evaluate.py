import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from utils.regression_loss import weighted_mse_loss, masked_f1_loss, reverse_log_transform, log_transform
from utils.dice_score import dice_coeff, multiclass_dice_coeff
from utils.df_loss import df_in_neighbor_loss, l1_loss_fn, denormalize_df, mae_loss
from utils.utils import downsample_torch_mask

@torch.inference_mode()
def evaluate(net, dataloader, device, amp, use_depth=False,
             only_depth=False, 
             use_mono_depth=False, head_mode="segmentation",
             reg_ds_factor=1.0):
    net.eval()
    num_val_batches = len(dataloader)

    # loss_fn_rg = masked_f1_loss
    loss_fn_rg = mae_loss
    loss_fn_df = mae_loss

    dice_score = 0
    reg_loss = 0
    df_loss = 0

    autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'

    with torch.autocast(autocast_device, enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image = batch['image']
            interest = batch['interest']   # [B, 1, H, W] IDP map [0, 1]
            valid = batch['valid']         # [B, 1, H, W] bool
            label_mask = batch['label_mask']  # [B, H, W] int64

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            interest = interest.to(device=device, dtype=torch.float32)
            valid = valid.to(device=device)
            label_mask = label_mask.to(device=device, dtype=torch.long)

            ds_true_df = interest.squeeze(1)  # [B, H, W]

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
                df_pred = denormalize_df(df_pred, df_neighborhood=10)
                df_loss += loss_fn_df(df_pred.float().squeeze(1), ds_true_df)
                # mask_true_log = log_transform(mask_true)
                reg_loss += loss_fn_rg(mask_pred.squeeze(), mask_true.float())

            elif head_mode == "df_seg":
                df_pred, masks_pred = net(image)
                # Masked L1 on the [0,1] IDP interest map
                valid_flat = valid.squeeze(1)  # [B, H, W]
                df_loss += (F.l1_loss(df_pred.squeeze(1), ds_true_df, reduction='none')
                            * valid_flat.float()).sum() / (valid_flat.sum() + 1e-6)
                # Dice score on segmentation head
                mask_true_oh = F.one_hot(label_mask, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred_oh = F.one_hot(masks_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                valid_mask = (label_mask != 0).unsqueeze(1).repeat(1, net.n_classes, 1, 1)
                dice_score += multiclass_dice_coeff(mask_pred_oh, mask_true_oh, valid_mask, reduce_batch_first=True)

            elif head_mode == "gmm":
                heatmap_pred, depth_pred = net(image)   # [B,1,H,W], [B,1,H,W]

                # Heatmap MAE (logged as df_loss for W&B compatibility)
                df_loss += F.l1_loss(heatmap_pred.squeeze(1), interest.squeeze(1))

                # Masked depth MAE (logged as reg_loss)
                depth_map = batch['depth_map'].to(device=device, dtype=torch.float32)
                depth_mask_flat = valid.squeeze(1).bool()   # [B, H, W]
                if depth_mask_flat.sum() > 0:
                    reg_loss += F.l1_loss(
                        depth_pred.squeeze(1)[depth_mask_flat],
                        depth_map.squeeze(1)[depth_mask_flat],
                    )

            elif head_mode == "detr":
                uv_pred, z_pred, conf_pred = net(image)   # (B,N,2), (B,N,1), (B,N,1)
                B, N = uv_pred.shape[:2]
                H_img, W_img = image.shape[2], image.shape[3]

                batch_conf_loss = torch.tensor(0.0, device=device)
                batch_match_loss = torch.tensor(0.0, device=device)

                for b in range(B):
                    gmm_b      = batch['gmm'][b]
                    centres_2d = gmm_b.get("centres_2d", None)
                    centres_3d = gmm_b.get("centres_3d", None)

                    conf_targets = torch.zeros(N, device=device)

                    if centres_2d is None or len(centres_2d) == 0:
                        batch_conf_loss = batch_conf_loss + F.binary_cross_entropy(
                            conf_pred[b].squeeze(-1), conf_targets
                        )
                        continue

                    K = len(centres_2d)
                    gt_uv = torch.tensor(centres_2d, device=device, dtype=torch.float32)
                    gt_uv[:, 0] = gt_uv[:, 0] / W_img
                    gt_uv[:, 1] = gt_uv[:, 1] / H_img
                    gt_z = torch.zeros(K, device=device, dtype=torch.float32)
                    if centres_3d is not None:
                        gt_z = torch.tensor(centres_3d[:, 2], device=device, dtype=torch.float32)

                    uv_b = uv_pred[b]
                    z_b  = z_pred[b].squeeze(-1)

                    cost_uv = torch.cdist(uv_b, gt_uv, p=1)
                    cost_z  = torch.abs(z_b.unsqueeze(1) - gt_z.unsqueeze(0))
                    cost    = (cost_uv + 0.1 * cost_z).cpu().numpy()

                    row_ind, col_ind = linear_sum_assignment(cost)

                    if len(row_ind) > 0:
                        batch_match_loss = batch_match_loss + (
                            F.l1_loss(uv_b[row_ind], gt_uv[col_ind])
                            + 0.1 * F.l1_loss(z_b[row_ind], gt_z[col_ind])
                        )

                    conf_targets[row_ind] = 1.0
                    batch_conf_loss = batch_conf_loss + F.binary_cross_entropy(
                        conf_pred[b].squeeze(-1), conf_targets
                    )

                # store: conf → dice_score slot, match uv+z → reg_loss slot
                dice_score += batch_conf_loss / B
                reg_loss   += batch_match_loss / B

    net.train()
    avg_dice_score = dice_score / num_val_batches if dice_score != 0 else 0
    avg_reg_loss = reg_loss / num_val_batches if reg_loss != 0 else 0
    avg_df_loss = df_loss / num_val_batches if df_loss != 0 else 0

    return avg_dice_score, avg_reg_loss, avg_df_loss