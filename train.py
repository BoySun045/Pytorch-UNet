import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluate import evaluate
from unet import TwoHeadUnet
from utils.dice_score import dice_loss
import datetime
import os
import sys
sys.path.insert(0, "/cluster/project/cvg/students/shangwu/dpt_distillation_repo")
from data import InterestDataset, collate_interest as collate_idp
from gmm_data import GmmDataset, collate_gmm


dir_img = Path("/cluster/project/cvg/students/shangwu/GEN3C/assets/diffusion/dataset_all")
dir_gmm = Path("/cluster/project/cvg/students/shangwu/FrontierNet/gmm_output")
dit_features_dir = Path("/cluster/project/cvg/students/shangwu/GEN3C/features_analysis/dit_features")
idp_cache_dir = Path(os.environ.get("SCRATCH", "/tmp")) / "cache_idp_features"
dir_checkpoint = Path("/cluster/project/cvg/students/shangwu/Pytorch-UNet/checkpoints") / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def plot_images(wandb_rgb, wandb_depth,
                 true_masks, label_mask,
                 wandb_mask_pred, binary_mask, 
                 wandb_df_pred, ds_true_df,
                 error_map, use_depth):
    num_cols = 5 
    fig, axes = plt.subplots(2, num_cols, figsize=(20, 8))
    
    # RGB image
    axes[0, 0].imshow(np.transpose(wandb_rgb[0].cpu().detach().numpy(), (1, 2, 0)))
    axes[0, 0].set_title('RGB Image')
    axes[0, 0].axis('on')

    idx_offset = 1
    # Depth image (if applicable)
    axes[0, 1].imshow(wandb_depth[0].cpu().detach().numpy(), cmap='gray')
    title = 'Depth Image ' if use_depth else 'Depth Image(Not Used)'
    axes[0, 1].set_title(title)
    axes[0, 1].axis('on')
    idx_offset += 1

    # True mask as heatmap
    axes[0, idx_offset].imshow(true_masks[0].cpu().detach().numpy(), cmap='viridis')
    axes[0, idx_offset].set_title('True weights Mask, max: ' + str(true_masks[0].max().item())) 
    axes[0, idx_offset].axis('on')

    # True binary mask
    axes[0, idx_offset + 1].imshow(label_mask[0].cpu().detach().numpy(), cmap='hot')
    axes[0, idx_offset + 1].set_title('True Label Mask, max: ' + str(label_mask[0].max().item()))
    plt.colorbar(axes[0, idx_offset + 1].imshow(label_mask[0].cpu().detach().numpy(), cmap='hot'), ax=axes[0, idx_offset + 1])
    axes[0, idx_offset + 1].axis('on')

    # Predicted mask as heatmap
    axes[0, idx_offset + 2].imshow(wandb_mask_pred[0].cpu().detach().numpy(), cmap='hot')
    axes[0, idx_offset + 2].set_title('Predicted Mask, max: ' + str(wandb_mask_pred[0].max().item()))
    plt.colorbar(axes[0, idx_offset + 2].imshow(wandb_mask_pred[0].cpu().detach().numpy(), cmap='hot'), ax=axes[0, idx_offset + 2])
    axes[0, idx_offset + 2].axis('on')

    # Error map as heatmap
    axes[1, 0].imshow(error_map[0].cpu().detach().numpy(), cmap='viridis')
    axes[1, 0].set_title('Error Map (Heatmap)')
    axes[1, 0].axis('on')

    # Predicted binary mask
    axes[1, 1].imshow(binary_mask[0].cpu().detach().numpy(), cmap='gray')
    axes[1, 1].set_title('Predicted Binary Mask')
    axes[1, 1].axis('on')

    # True distance field
    axes[1, 2].imshow(ds_true_df[0].cpu().detach().numpy(), cmap='viridis')
    axes[1, 2].set_title('True Distance Field')
    axes[1, 2].axis('on')

    # Predicted distance field
    axes[1, 3].imshow(wandb_df_pred[0].cpu().detach().numpy(), cmap='viridis')
    axes[1, 3].set_title('Predicted Distance Field')
    axes[1, 3].axis('on')
    
    # Remove any empty axes
    for ax in axes.ravel():
        if not ax.has_data():
            ax.axis('off')

    # Convert plot to an image
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return img_array

def log_images(experiment, optimizer, 
               val_score_cl, val_score_rg, val_score_df, 
               wandb_rgb, wandb_depth,
               true_masks, label_mask, 
               wandb_mask_pred, binary_mask,
               wandb_df_pred, ds_true_df, 
               global_step, epoch, histograms, use_depth):

    # Calculate the error map
    error_map = torch.abs(ds_true_df - wandb_df_pred).cpu().detach()
    
    combined_image = plot_images(wandb_rgb, wandb_depth, 
                                 true_masks, label_mask,
                                   wandb_mask_pred, binary_mask,
                                   wandb_df_pred, ds_true_df, 
                                   error_map, use_depth)
    
    try:
        experiment.log({
            'learning rate': optimizer.param_groups[0]['lr'],
            'validation avg score classification': val_score_cl,
            'validation avg score regression': val_score_rg,
            'validation avg score distance field': val_score_df,
            'combined images': wandb.Image(combined_image),
            'step': global_step,
            'epoch': epoch,
            **histograms
        })
    except Exception as e:
        print(f"Failed to log to Weights and Biases: {e}")


        
def hungarian_match_loss(
    uv_pred, z_pred, conf_pred, weight_pred, occ_pred,
    gmm_list, H, W, device,
    z_weight=0.1, weight_loss_weight=0.1, occ_loss_weight=0.5,
):
    """Per-batch DETR Hungarian matching loss.

    Args:
        uv_pred          : (B, N, 2)  normalised predicted coords [0, 1]
        z_pred           : (B, N, 1)  predicted depth [m]
        conf_pred        : (B, N, 1)  predicted confidence [0, 1]
        weight_pred      : (B, N, 1)  predicted GMM component weight (softplus)
        occ_pred         : (B, N, 1)  predicted occlusion probability [0, 1]
        gmm_list         : list of B dicts with 'centres_2d', 'centres_3d',
                           'weights', and optionally 'occlusion'
        H, W             : original image height / width (for GT normalisation)
        z_weight         : weight of z term in bipartite matching cost and regression loss
        weight_loss_weight: weight of GMM component weight L1 loss term
        occ_loss_weight  : weight of occlusion BCE loss term

    Returns:
        match_loss, conf_loss, occ_loss  (scalars, averaged over batch)
    """
    B, N = uv_pred.shape[:2]
    total_match = torch.tensor(0.0, device=device)
    total_conf  = torch.tensor(0.0, device=device)
    total_occ   = torch.tensor(0.0, device=device)

    for b in range(B):
        gmm_b      = gmm_list[b]
        centres_2d = gmm_b.get("centres_2d", None)   # (K, 2) numpy, pixel (u, v)
        centres_3d = gmm_b.get("centres_3d", None)   # (K, 3) numpy, camera [m]
        gt_weights = gmm_b.get("weights",    None)   # (K,)   numpy
        gt_occ_np  = gmm_b.get("occlusion",  None)   # (K,)   bool numpy or None

        conf_targets = torch.zeros(N, device=device)

        if centres_2d is None or len(centres_2d) == 0:
            total_conf = total_conf + F.binary_cross_entropy(
                conf_pred[b].squeeze(-1), conf_targets
            )
            continue

        K = len(centres_2d)

        # Normalise GT uv to [0, 1]
        gt_uv = torch.tensor(centres_2d, device=device, dtype=torch.float32)
        gt_uv[:, 0] /= W
        gt_uv[:, 1] /= H

        gt_z = torch.zeros(K, device=device, dtype=torch.float32)
        if centres_3d is not None:
            gt_z = torch.tensor(centres_3d[:, 2], device=device, dtype=torch.float32)

        # --- Build cost matrix (no grad) -----------------------------------
        uv_b = uv_pred[b]              # (N, 2)
        z_b  = z_pred[b].squeeze(-1)   # (N,)

        with torch.no_grad():
            cost_uv = torch.cdist(uv_b.detach(), gt_uv, p=1)                    # (N, K)
            cost_z  = torch.abs(z_b.detach().unsqueeze(1) - gt_z.unsqueeze(0))  # (N, K)
            cost    = cost_uv + z_weight * cost_z                                # (N, K)

        row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())

        # --- Regression loss on matched slots (uv + z + weight) -----------
        if len(row_ind) > 0:
            reg = F.l1_loss(uv_b[row_ind], gt_uv[col_ind])
            reg = reg + z_weight * F.l1_loss(z_b[row_ind], gt_z[col_ind])

            if gt_weights is not None:
                gt_w_t = torch.tensor(
                    gt_weights[col_ind], device=device, dtype=torch.float32
                )
                reg = reg + weight_loss_weight * F.l1_loss(
                    weight_pred[b][row_ind].squeeze(-1), gt_w_t
                )

            total_match = total_match + reg

        # --- Confidence loss: matched → 1, unmatched → 0 ------------------
        conf_targets[row_ind] = 1.0
        total_conf = total_conf + F.binary_cross_entropy(
            conf_pred[b].squeeze(-1), conf_targets
        )

        # --- Occlusion loss on matched slots (only when GT available) ------
        if gt_occ_np is not None and len(row_ind) > 0:
            gt_occ_t = torch.tensor(
                gt_occ_np[col_ind].astype(np.float32), device=device
            )
            total_occ = total_occ + occ_loss_weight * F.binary_cross_entropy(
                occ_pred[b][row_ind].squeeze(-1), gt_occ_t
            )

    return total_match / B, total_conf / B, total_occ / B


def plot_detr_predictions(
    rgb, uv_pred, z_pred, conf_pred, weight_pred, occ_pred,
    gmm_list, H, W, conf_thresh=0.3,
):
    """Overlay predicted and GT frontier centres on the first batch image.

    GT centres: lime × = visible, red × = occluded.  Label shows z and weight.
    Predicted slots above conf_thresh: colour = occlusion prob (green→red),
        label shows z, confidence, predicted weight and occlusion probability.
    """
    img_np = rgb[0].cpu().permute(1, 2, 0).float().numpy().clip(0, 1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.imshow(img_np)

    # --- GT centres --------------------------------------------------------
    gmm_b      = gmm_list[0]
    centres_2d = gmm_b.get("centres_2d", None)
    centres_3d = gmm_b.get("centres_3d", None)
    gt_weights = gmm_b.get("weights",    None)
    gt_occ     = gmm_b.get("occlusion",  None)
    if centres_2d is not None and len(centres_2d) > 0:
        for k, (u, v) in enumerate(centres_2d):
            is_occ   = bool(gt_occ[k]) if gt_occ is not None else False
            gt_color = "red" if is_occ else "lime"
            ax.scatter(u, v, s=120, c=gt_color, marker="x", linewidths=2)
            gt_z_val = float(centres_3d[k, 2]) if centres_3d is not None else float("nan")
            gt_w_val = float(gt_weights[k]) if gt_weights is not None else float("nan")
            occ_str  = " OCC" if is_occ else ""
            ax.annotate(f"GT z={gt_z_val:.1f}m w={gt_w_val:.2f}{occ_str}",
                        xy=(u, v), xytext=(4, -12), textcoords="offset points",
                        color=gt_color, fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.4))

    # --- Predicted slots above threshold -----------------------------------
    uv_np     = uv_pred[0].cpu().detach().numpy()               # (N, 2)
    z_np      = z_pred[0].squeeze(-1).cpu().detach().numpy()    # (N,)
    conf_np   = conf_pred[0].squeeze(-1).cpu().detach().numpy() # (N,)
    weight_np = weight_pred[0].squeeze(-1).cpu().detach().numpy() # (N,)
    occ_np    = occ_pred[0].squeeze(-1).cpu().detach().numpy()  # (N,)
    for n in range(len(uv_np)):
        if conf_np[n] >= conf_thresh:
            u_px = uv_np[n, 0] * W
            v_px = uv_np[n, 1] * H
            # colour interpolates green (not occluded) → red (occluded)
            slot_color = (float(occ_np[n]), 1.0 - float(occ_np[n]), 0.0)
            ax.scatter(u_px, v_px, s=80 * conf_np[n], color=slot_color, marker="o",
                       alpha=float(conf_np[n]))
            ax.annotate(
                f"z={z_np[n]:.1f}m c={conf_np[n]:.2f}\nw={weight_np[n]:.2f} occ={occ_np[n]:.2f}",
                xy=(u_px, v_px), xytext=(4, 6), textcoords="offset points",
                color="white", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.4),
            )

    ax.axis("off")
    ax.set_title(
        "DETR: GT limex=visible redx=occluded | pred colour=occ prob (green→red)"
    )

    plt.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img_array = img_array[:, :, :3]
    plt.close(fig)
    return img_array


def plot_detr_3d(uv_pred, z_pred, conf_pred, gmm_list, H, W, conf_thresh=0.3):
    """3-D matplotlib figure: GT GMM ellipsoids + DETR predicted centres.

    GT components (warm colourmap spheres + wireframe ellipsoids).
    Predicted slots above conf_thresh (cyan triangles), back-projected to 3-D
    using the GmmDataset camera intrinsics.
    """
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (registers 3d projection)

    # Camera intrinsics matching GmmDataset._CAM_F / _CAM_W / _CAM_H
    CAM_F = 300.0
    fx = CAM_F * (W / 720.0)
    fy = CAM_F * (H / 544.0)
    cx, cy = W / 2.0, H / 2.0

    gmm_b       = gmm_list[0]
    centres_3d  = gmm_b.get("centres_3d",  None)   # (K, 3)
    covariances = gmm_b.get("covariances", None)    # (K, 3, 3)
    weights     = gmm_b.get("weights",     None)    # (K,)

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    # Unit sphere vertices (reused for every ellipsoid)
    _u  = np.linspace(0, 2 * np.pi, 20)
    _v  = np.linspace(0,     np.pi, 20)
    sx  = np.outer(np.cos(_u), np.sin(_v))
    sy  = np.outer(np.sin(_u), np.sin(_v))
    sz  = np.outer(np.ones_like(_u), np.cos(_v))
    sphere_pts = np.stack([sx.ravel(), sy.ravel(), sz.ravel()], axis=1)  # (400, 3)

    # --- GT GMM components ------------------------------------------------
    if centres_3d is not None and len(centres_3d) > 0:
        K      = len(centres_3d)
        w_norm = weights / (weights.max() + 1e-8) if weights is not None else np.ones(K)
        colors = plt.cm.hot(np.linspace(0.3, 1.0, K))

        for i in range(K):
            c   = centres_3d[i]
            col = colors[i]

            ax.scatter(*c,
                       s=100 * (0.5 + 0.5 * w_norm[i]),
                       c=[col], marker="o",
                       label=f"GT #{i}  z={c[2]:.1f}m  w={w_norm[i]:.2f}")

            if covariances is not None:
                cov              = covariances[i]
                eigvals, eigvecs = np.linalg.eigh(cov)
                eigvals          = np.clip(eigvals, 1e-8, None)
                scale            = np.sqrt(eigvals)

                R = eigvecs.copy()
                if np.linalg.det(R) < 0:
                    R[:, 0] *= -1

                ell_pts = (R @ (sphere_pts * scale).T).T + c  # (400, 3)
                ex = ell_pts[:, 0].reshape(20, 20)
                ey = ell_pts[:, 1].reshape(20, 20)
                ez = ell_pts[:, 2].reshape(20, 20)
                ax.plot_wireframe(ex, ey, ez,
                                  color=col, alpha=0.2, linewidth=0.5)

    # --- Back-project predicted slots to 3-D -----------------------------
    uv_np   = uv_pred[0].cpu().detach().numpy()           # (N, 2)
    z_np    = z_pred[0].squeeze(-1).cpu().detach().numpy()   # (N,)
    conf_np = conf_pred[0].squeeze(-1).cpu().detach().numpy()  # (N,)

    pred_pts = []
    for n in range(len(uv_np)):
        if conf_np[n] >= conf_thresh:
            u_px = uv_np[n, 0] * W
            v_px = uv_np[n, 1] * H
            z    = z_np[n]
            x    = (u_px - cx) * z / fx
            y    = (v_px - cy) * z / fy
            pred_pts.append((x, y, z))

    if pred_pts:
        xs, ys, zs = zip(*pred_pts)
        ax.scatter(xs, ys, zs, s=80, c="cyan", marker="^",
                   alpha=0.9, label=f"Pred  conf≥{conf_thresh:.1f}")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("DETR 3-D: GT ellipsoids (warm) · predictions (cyan ▲)")
    ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img_array = img_array[:, :, :3]
    plt.close(fig)
    return img_array


def plot_detr_depth(depth_pred, depth_gt):
    """1×3 figure: GT depth | predicted depth | absolute error.

    Args:
        depth_pred : [B, 1, H, W] float32 tensor  — aux depth head output
        depth_gt   : [B, 1, H, W] float32 tensor  — scene depth from depth_dir

    Only the first batch element is shown.
    """
    def to_np(t):
        return t[0, 0].cpu().detach().float().numpy()

    gt_np   = to_np(depth_gt)
    pred_np = to_np(depth_pred)
    err_np  = np.abs(pred_np - gt_np)

    valid = gt_np > 0.0   # mask out invalid (zero) depth pixels

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    vmin = float(gt_np[valid].min()) if valid.any() else 0.0
    vmax = float(gt_np[valid].max()) if valid.any() else 1.0

    ax = axes[0]
    disp = np.where(valid, gt_np, np.nan)
    im = ax.imshow(disp, cmap="plasma", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.03, label="m")
    ax.set_title(f"GT depth [m]  (valid={valid.mean():.2%})")
    ax.axis("off")

    ax = axes[1]
    disp = np.where(valid, pred_np, np.nan)
    im = ax.imshow(disp, cmap="plasma", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.03, label="m")
    mae = float(err_np[valid].mean()) if valid.any() else float("nan")
    ax.set_title(f"Pred depth [m]  (MAE={mae:.3f}m)")
    ax.axis("off")

    ax = axes[2]
    disp = np.where(valid, err_np, np.nan)
    im = ax.imshow(disp, cmap="hot", vmin=0)
    fig.colorbar(im, ax=ax, fraction=0.03, label="m")
    ax.set_title("|error| [m]")
    ax.axis("off")

    plt.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img_array = img_array[:, :, :3]
    plt.close(fig)
    return img_array


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        gradient_clipping: float = 1.0,
        use_depth: bool = False,
        only_depth: bool = False,
        use_mono_depth: bool = False,
        reg_loss_weight: float = 1.0,
        head_mode: str = 'df_seg',
        dataset_portion: float = 1.0,
        lr_decay: bool = True,
        reg_ds_factor = 1.0,
        aux_depth_weight: float = 0.5,
):
    # 1. Create dataset
    data_augmentation = False

    if head_mode == "detr":
        dataset = GmmDataset(
            gmm_dir="/cluster/project/cvg/students/shangwu/Pytorch-UNet/gmm_dummy/gmm",
            image_dir="/cluster/project/cvg/students/shangwu/Pytorch-UNet/gmm_dummy/image",
            depth_dir="/cluster/project/cvg/students/shangwu/Pytorch-UNet/gmm_dummy/depth",
            image_size=(544, 720),
            augment=data_augmentation,
        )
        collate_fn = collate_gmm
    else:
        # Build dataset from IDP cache
        dataset = InterestDataset(
            image_dir=str(dir_img),
            dit_features_dir=str(dit_features_dir),
            image_size=(544, 720),
            idp_cache_dir=str(idp_cache_dir),
            num_classes=model.n_classes,
            seg_bin_edges=(0.05, 0.15, 0.3, 0.45, 0.6, 0.8),
            augment=data_augmentation,
        )
        collate_fn = collate_idp

    # 2. Subset the dataset
    total_size = int(len(dataset) * dataset_portion)
    dataset, _ = random_split(dataset, [total_size, len(dataset) - total_size], generator=torch.Generator().manual_seed(0))

    # 3. Split into train / validation partitions
    n_val = int(total_size * val_percent)
    n_train = total_size - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    print(f"Train size: {n_train}, Validation size: {n_val}")

    # 4. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=16, pin_memory=True, collate_fn=collate_fn)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net-resnet-v3', entity='ftnet-wm', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs,
             batch_size=batch_size,
             learning_rate=learning_rate,
             val_percent=val_percent,
             save_checkpoint=save_checkpoint,
             dataset_portion=dataset_portion,
             do_data_augmentation=data_augmentation,
             use_depth=use_depth,
             only_depth=only_depth,
             use_mono_depth=use_mono_depth,
             regloss_weight=reg_loss_weight,
             lr_decay=lr_decay,
             regression_downsample_factor=reg_ds_factor,
             amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if lr_decay:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.5, min_lr=5e-6)  # goal: maximize score
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10000000, factor=0.5, min_lr=5e-5)  # goal: minimize loss
    
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # 5. set up losses
    # For df_seg: CE+Dice on label_mask; masked L1 on the [0,1] interest map
    loss_fn_cl = nn.CrossEntropyLoss(ignore_index=0)

    global_step = 0
    reg_loss_weight = reg_loss_weight

    # 6. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for _, batch in enumerate(train_loader):

                images = batch['image'].to(
                    device=device, dtype=torch.float32,
                    memory_format=torch.channels_last,
                )

                # IDP-only keys — only available outside detr mode
                if head_mode != "detr":
                    interest   = batch['interest'].to(device=device, dtype=torch.float32)
                    valid      = batch['valid'].to(device=device)
                    label_mask = batch['label_mask'].to(device=device, dtype=torch.long)
                    ds_true_df    = interest.squeeze(1)   # [B, H, W]
                    ds_true_masks = interest.squeeze(1)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):

                    if head_mode == "df_seg":
                        df_pred, masks_pred = model(images)
                        valid_flat = valid.squeeze(1)  # [B, H, W]
                        df_loss = (F.l1_loss(df_pred.squeeze(1), interest.squeeze(1), reduction='none')
                                   * valid_flat.float()).sum() / (valid_flat.sum() + 1e-6)
                        class_loss = loss_fn_cl(masks_pred, label_mask)
                        valid_mask = (label_mask != 0).unsqueeze(1).repeat(1, model.n_classes, 1, 1)
                        class_loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(label_mask, model.n_classes).permute(0, 3, 1, 2).float(),
                            valid_mask,
                            multiclass=True if model.n_classes > 1 else False
                        )
                        if global_step > 5000:
                            reg_loss_weight = max(2.0, reg_loss_weight * 0.99)
                        loss = reg_loss_weight * df_loss + class_loss
                        reg_loss = None

                    elif head_mode == "detr":
                        # (B,N,2)  (B,N,1)  (B,N,1)  (B,N,1)  (B,N,1)  [B,1,H,W]|None
                        uv_pred, z_pred, conf_pred, weight_pred, occ_pred, depth_pred = model(images)

                        H_img, W_img = images.shape[2], images.shape[3]
                        reg_loss, class_loss, occ_loss = hungarian_match_loss(
                            uv_pred, z_pred, conf_pred, weight_pred, occ_pred,
                            batch['gmm'], H_img, W_img, device,
                            z_weight=0.1,
                        )
                        loss = reg_loss + reg_loss_weight * class_loss + occ_loss

                        # Auxiliary dense depth supervision (only when --aux_depth is set)
                        aux_depth_loss = torch.tensor(0.0, device=device)
                        if depth_pred is not None and batch['depth_map'] is not None:
                            depth_gt = batch['depth_map'].to(device=device, dtype=torch.float32)
                            # Align spatial size: resize pred to match GT if they differ
                            if depth_pred.shape[-2:] != depth_gt.shape[-2:]:
                                depth_pred_aligned = F.interpolate(
                                    depth_pred, size=depth_gt.shape[-2:], mode="bilinear", align_corners=False
                                )
                            else:
                                depth_pred_aligned = depth_pred
                            valid_px = depth_gt.squeeze(1) > 0.0   # [B, H, W]
                            if valid_px.any():
                                aux_depth_loss = F.huber_loss(
                                    depth_pred_aligned.squeeze(1)[valid_px],
                                    depth_gt.squeeze(1)[valid_px],
                                    delta=1.0,
                                )
                            loss = loss + aux_depth_weight * aux_depth_loss
                        df_loss = None

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                if head_mode == "detr":
                    experiment.log({
                        'train loss total':       loss.item(),
                        'train loss match uv+z':  reg_loss.item(),
                        'train loss conf':        class_loss.item(),
                        'train loss occlusion':   occ_loss.item(),
                        'train loss aux depth':   aux_depth_loss.item(),
                        'step': global_step,
                        'epoch': epoch,
                    })
                else:
                    experiment.log({
                        'train loss total': loss.item(),
                        'train loss regression': reg_loss.item() if reg_loss is not None else 0.0,
                        'train loss classification': class_loss.item() if class_loss is not None else 0.0,
                        'train loss df': df_loss.item() if df_loss is not None else 0.0,
                        'step': global_step,
                        'epoch': epoch,
                    })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                # division_step = (n_train // (10 * batch_size))
                division_step = 10
                if division_step > 0 and global_step % division_step == 0:
                    histograms = {}
                    val_score_cl, val_score_rg, val_score_df = evaluate(model, val_loader, device, amp,
                                                          use_depth=use_depth, 
                                                          only_depth = only_depth,
                                                          use_mono_depth = use_mono_depth,
                                                          head_mode = head_mode,
                                                          reg_ds_factor=reg_ds_factor)

                    if head_mode == "df_seg":
                        scheduler.step(1 - reg_loss_weight * val_score_df + val_score_cl)
                        wandb_df_pred = df_pred.squeeze(1)          # [B, H, W] already [0,1]
                        softmax_pred = F.softmax(masks_pred, dim=1)
                        max_class_pred = torch.argmax(softmax_pred, dim=1, keepdim=True)
                        wandb_mask_pred = max_class_pred.squeeze(1) * (label_mask != 0)
                        binary_mask = torch.zeros_like(interest.squeeze(1))

                    elif head_mode == "detr":
                        scheduler.step(-(val_score_rg + reg_loss_weight * val_score_cl))
                        H_img, W_img = images.shape[2], images.shape[3]
                        overlay = plot_detr_predictions(
                            images[:, :3], uv_pred, z_pred, conf_pred,
                            weight_pred, occ_pred,
                            batch['gmm'], H_img, W_img,
                        )
                        vis_3d = plot_detr_3d(
                            uv_pred, z_pred, conf_pred, batch['gmm'], H_img, W_img
                        )
                        log_dict = {
                            'learning rate':          optimizer.param_groups[0]['lr'],
                            'val match uv+z (MAE)':   val_score_rg,
                            'val conf loss':          val_score_cl,
                            'train loss occlusion':   occ_loss.item(),
                            'predictions overlay':    wandb.Image(overlay),
                            '3d gmm vis':             wandb.Image(vis_3d),
                            'step':                   global_step,
                            'epoch':                  epoch,
                            **histograms,
                        }
                        if depth_pred is not None and batch['depth_map'] is not None:
                            depth_gt = batch['depth_map'].to(device=device, dtype=torch.float32)
                            depth_pred_vis = F.interpolate(depth_pred, size=depth_gt.shape[-2:], mode="bilinear", align_corners=False) if depth_pred.shape[-2:] != depth_gt.shape[-2:] else depth_pred
                            depth_vis = plot_detr_depth(depth_pred_vis, depth_gt)
                            log_dict['aux depth GT vs pred'] = wandb.Image(depth_vis)
                        try:
                            experiment.log(log_dict)
                        except Exception as e:
                            print(f"Failed to log to Weights and Biases: {e}")
                        continue   # skip generic log_images call below

                    logging.info(f'Validation Classification Dice score: {val_score_cl}')
                    logging.info(f'Validation Regression mse : {val_score_rg}')
                    logging.info(f'Validation distance field mse : {val_score_df}')

                    wandb_rgb = images[:, :3, :, :]
                    wandb_depth = interest.squeeze(1)   # show IDP map in place of depth slot

                    log_images(experiment, optimizer,
                                 val_score_cl, val_score_rg, val_score_df,
                                 wandb_rgb, wandb_depth,
                                 ds_true_masks, label_mask,
                                 wandb_mask_pred, binary_mask,
                                 wandb_df_pred, ds_true_df,
                                 global_step, epoch, histograms, use_depth)


        if save_checkpoint and epoch % 2 == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            use_depth_str = 'depth' if use_depth else 'no_depth'
            reg_weights = str(reg_loss_weight)
            torch.save(state_dict, str(dir_checkpoint / f'CP_epoch{epoch}_{use_depth_str}_{only_depth}_{reg_weights}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=2e-6,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=0.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--reg_loss_weight', '-rw', type=float, default=1.0, help='Weight of regression loss')
    parser.add_argument('--use_depth','-ud', action='store_true', default=False, help='Use depth image')
    parser.add_argument('--only_depth','-od', action='store_true', default=False, help='Only use depth image')
    parser.add_argument('--use_mono_depth','-umd', action='store_true', default=False, help='Use mono depth image')
    parser.add_argument('--head_mode', type=str, default='df_seg', help='df_seg, gmm, or detr')
    parser.add_argument('--regression_downsample_factor','-rdf', type=float, default=1.0, help='Downsample factor for regression head')
    parser.add_argument('--num_queries', '-nq', type=int, default=10, help='Number of DETR slot queries (detr mode only)')
    parser.add_argument('--aux_depth', action='store_true', default=False,
                        help='Add auxiliary UNet-decoder depth head in detr mode')
    parser.add_argument('--aux_depth_weight', type=float, default=0.3,
                        help='Weight of the auxiliary depth Huber loss (detr mode only)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    head_mode = args.head_mode
    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_channels=4 for RGB-D images
    # n_classes is the number of probabilities you want to get per pixel
    if args.use_depth and not args.only_depth:
        print("Using RGB-D images")
        model = TwoHeadUnet(classes=args.classes,
                            in_channels=4,
                            head_config=head_mode,
                            regression_downsample_factor=args.regression_downsample_factor,
                            num_queries=args.num_queries,
                            detr_aux_depth=args.aux_depth)

    if args.use_depth and args.only_depth:
        model = TwoHeadUnet(classes=args.classes,
                            in_channels=1,
                            head_config=head_mode,
                            regression_downsample_factor=args.regression_downsample_factor,
                            num_queries=args.num_queries,
                            detr_aux_depth=args.aux_depth)

    if not args.use_depth:
        model = TwoHeadUnet(classes=args.classes,
                            in_channels=3,
                            head_config=head_mode,
                            regression_downsample_factor=args.regression_downsample_factor,
                            num_queries=args.num_queries,
                            detr_aux_depth=args.aux_depth)
        
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n')
                #  f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        state_dict.pop('mask_values', None)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            use_depth=args.use_depth,
            only_depth=args.only_depth,
            use_mono_depth = args.use_mono_depth,
            reg_loss_weight=args.reg_loss_weight,
            head_mode=head_mode,
            weight_decay=1e-7,
            reg_ds_factor=args.regression_downsample_factor,
            aux_depth_weight=args.aux_depth_weight,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            use_depth=args.use_depth,
            only_depth=args.only_depth,
            use_mono_depth = args.use_mono_depth,
            reg_loss_weight=args.reg_loss_weight,
            head_mode=head_mode,
            weight_decay=1e-7,
            reg_ds_factor=args.regression_downsample_factor,
            aux_depth_weight=args.aux_depth_weight,
        )
