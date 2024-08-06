import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
import matplotlib.pyplot as plt
import numpy as np

from evaluate import evaluate
from unet import UNet, UnetResnet, TwoHeadUnet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from utils.regression_loss import mse_loss, weighted_mse_loss, masked_f1_loss, weighted_huber_loss, reverse_log_transform
from utils.df_loss import df_in_neighbor_loss, df_normalized_loss_in_neighbor, l1_loss_fn, denormalize_df
from utils.utils import downsample_torch_mask
from torchvision.utils import save_image
import datetime 


dir_path = Path("/mnt/boysunSSD/Actmap_v2_mini")
# dir_path = Path("/cluster/project/cvg/boysun/Actmap_v3")  # actmap_v3 is the one after data balancing cleaning
# dir_path = Path("/cluster/project/cvg/boysun/Actmap_v2_mini")
# dir_path = Path("/cluster/project/cvg/boysun/one_image_dataset_3")
# dir_path = Path("/mnt/boysunSSD//one_image_dataset_3")
dir_img = Path(dir_path / 'image/')
dir_mask = Path(dir_path / 'weighted_mask/')
dir_checkpoint = Path(dir_path / 'checkpoints' / datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
dir_debug = Path(dir_path / 'debug/')
dir_depth = Path(dir_path / 'depth/')

# make debug directory
dir_debug.mkdir(parents=True, exist_ok=True)

def save_debug_images(batch, epoch, batch_idx, prefix='train', num_images=5):
    """
    Saves a set of images, masks, and optionally depth maps from a batch for debugging.

    Args:
        batch (dict): The current batch of data containing 'image', 'mask', and optionally 'depth'.
        epoch (int): Current epoch number for naming.
        batch_idx (int): Current batch index for naming.
        prefix (str): Prefix for the filenames to indicate training or validation phase.
        num_images (int): Number of images to save from the batch.
    """
    images, masks = batch['image'][:num_images], batch['mask'][:num_images]
    depths = batch['depth'][:num_images] if 'depth' in batch else None

    for i in range(num_images):
        # save images and masks under the dir_debug directory
        img_path = f'{dir_debug}/{prefix}_epoch{epoch}_batch{batch_idx}_img{i}.jpg'
        mask_path = f'{dir_debug}/{prefix}_epoch{epoch}_batch{batch_idx}_mask{i}.jpg'
        save_image(images[i], img_path)
        save_image(masks[i], mask_path)

        if depths is not None:
            depth_path = f'{dir_debug}/{prefix}_epoch{epoch}_batch{batch_idx}_depth{i}.png'
            save_image(depths[i], depth_path)


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
    axes[0, idx_offset].set_title('True Mask (Heatmap)')
    axes[0, idx_offset].axis('on')

    # True binary mask
    axes[0, idx_offset + 1].imshow(label_mask[0].cpu().detach().numpy(), cmap='hot')
    axes[0, idx_offset + 1].set_title('True Label Mask')
    axes[0, idx_offset + 1].axis('on')

    # Predicted mask as heatmap
    axes[0, idx_offset + 2].imshow(wandb_mask_pred[0].cpu().detach().numpy(), cmap='viridis')
    axes[0, idx_offset + 2].set_title('Predicted Mask (Heatmap)')
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
    
    # error_map = torch.abs(true_masks - wandb_mask_pred).cpu().detach()
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
        momentum: float = 0.9,
        gradient_clipping: float = 1.0,
        use_depth: bool = False,
        use_mono_depth: bool = False,
        reg_loss_weight: float = 1.0,
        head_mode: str = 'segmentation',
        dataset_portion: float = 1.0,
        lr_decay: bool = True,
        reg_loss_type = 'huber',
        reg_loss_cal_inmask = True,
        log_transform = True,
        reg_ds_factor = 1.0
):
    # 1. Create dataset
    data_augmentation = True
    log_transform = log_transform

    # Always load depth, but only use it if set 
    try:
        dataset = CarvanaDataset(dir_img, dir_mask,dir_depth, 
                                 img_scale,
                                 gen_mono_depth = use_mono_depth,
                                 data_augmentation=data_augmentation, log_transform=log_transform)
        
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, dir_depth,
                                img_scale, 
                                gen_mono_depth = use_mono_depth,
                                data_augmentation=data_augmentation, log_transform=log_transform)

    # 2. Subset the dataset
    total_size = int(len(dataset) * dataset_portion)
    dataset, _ = random_split(dataset, [total_size, len(dataset) - total_size], generator=torch.Generator().manual_seed(0))

    # 3. Split into train / validation partitions
    n_val = int(total_size * val_percent)
    n_train = total_size - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    print(f"Train size: {n_train}, Validation size: {n_val}")

    # 4. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=32, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net-resnet-v3', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, 
             batch_size=batch_size, 
             learning_rate=learning_rate,
             val_percent=val_percent, 
             save_checkpoint=save_checkpoint,
             trainer_momentum=momentum,
             dataset_portion=dataset_portion,
             do_data_augmentation=data_augmentation,
             use_depth=use_depth,
             use_mono_depth = use_mono_depth,
             img_scale=img_scale,
             regloss_weight = reg_loss_weight,
             lr_decay = lr_decay,
             regression_loss_fn = reg_loss_type,
             regression_loss_cal_inmask = reg_loss_cal_inmask,
             log_transform = log_transform,
             regression_downsample_factor = reg_ds_factor, 
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
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    #use adam optimizer
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if lr_decay:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=15, factor=0.5, min_lr=5e-6)  # goal: maximize score
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10000000, factor=0.5, min_lr=5e-5)  # goal: minimize loss
    
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # 5. set up losses
    # loss_fn_rg = weighted_mse_loss
    # if reg_loss_type == 'l1_inv':
    #     loss_fn_rg = weighted_l1_inverse_loss
    # else:
    #     loss_fn_rg = weighted_huber_loss

    loss_fn_rg = masked_f1_loss
    pos_weight = torch.tensor([2.0]).to(device)
    loss_fn_cl = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # loss_fn_df = df_in_neighbor_loss
    loss_fn_df = df_normalized_loss_in_neighbor

    global_step = 0 
    class_loss_weight = 1.0
    reg_loss_weight = reg_loss_weight       
    # weight of regression loss really matters, 5.0 is a tested good one, if it's higher, e.g., 10.0, cls result becomes worse

    # 6. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for _, batch in enumerate(train_loader):

                true_masks, true_binary_masks = batch['mask'], batch['binary_mask']
                true_df = batch['df']
                images = batch['image']
                depth = batch['depth'] if not use_mono_depth else batch['mono_depth']
                label_mask = batch['label_mask']

                # assert images.shape[1] + depth.shape[1] == model.n_channels, \
                assert images.shape[1] + depth.shape[1] == 4, \
                    f'Network has been defined with {4} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                depth = depth.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

                images = torch.cat([images, depth], dim=1) if use_depth else images

                true_masks = true_masks.to(device=device, dtype=torch.float32)
                true_binary_masks = true_binary_masks.to(device=device, dtype=torch.float32)
                true_df = true_df.to(device=device, dtype=torch.float32)
                label_mask = label_mask.to(device=device, dtype=torch.float32)

                # do downsample for gt mask 
                ds_true_masks = downsample_torch_mask(true_masks, reg_ds_factor, "bilinear") if reg_ds_factor != 1.0 else true_masks
                ds_true_binary_masks = downsample_torch_mask(true_binary_masks, reg_ds_factor, "nearest") if reg_ds_factor != 1.0 else true_binary_masks
                ds_true_df = downsample_torch_mask(true_df, reg_ds_factor, "bilinear") if reg_ds_factor != 1.0 else true_df

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):

                    if head_mode == 'both':
                        binary_pred, masks_pred = model(images)

                        # reg_loss = loss_fn_rg(masks_pred.squeeze(1), true_masks.float(), true_binary_masks.float(), 
                        #                       increase_factor=5.0, avg_using_binary_mask=reg_loss_cal_inmask)

                        reg_loss = loss_fn_rg(masks_pred.squeeze(1), ds_true_masks.float(), ds_true_binary_masks.float(),
                                                increase_factor=5.0, avg_using_binary_mask=reg_loss_cal_inmask)

                        class_loss = loss_fn_cl(binary_pred.squeeze(1), true_binary_masks.float())
                        class_loss += dice_loss(F.sigmoid(binary_pred.squeeze(1)), true_binary_masks.float(), multiclass=False)
                        class_loss = class_loss_weight * class_loss
                        reg_loss = reg_loss_weight * reg_loss
                        loss = reg_loss + class_loss

                    elif head_mode == 'segmentation':
                        binary_pred = model(images)
                        class_loss = loss_fn_cl(binary_pred.squeeze(1), true_binary_masks.float())
                        class_loss += dice_loss(F.sigmoid(binary_pred.squeeze(1)), true_binary_masks.float(), multiclass=False)
                        loss =  class_loss

                        reg_loss = None
                        
                    elif head_mode == 'regression':
                        masks_pred = model(images)
                        # reg_loss = loss_fn_rg(masks_pred.squeeze(1), true_masks.float(), true_binary_masks.float(), 
                        #                       increase_factor=8.0, avg_using_binary_mask=False)

                        # wf loss
                        reg_loss = loss_fn_rg(masks_pred.squeeze(1), true_masks.float())
                        loss = reg_loss
                        
                        df_loss = None
                        class_loss = None
                    
                    elif head_mode == 'df':
                        df_pred = model(images)
                        df_loss = loss_fn_df(df_pred.squeeze(1), ds_true_df)
                        loss = df_loss
                        class_loss = None
                        reg_loss = None

                    elif head_mode == "df_wf":
                        df_pred, masks_pred = model(images)
                        print("true mask min max", true_masks.min(), true_masks.max())
                        reg_loss = loss_fn_rg(masks_pred.squeeze(1), true_masks.float(), df = ds_true_df)
                        df_loss = loss_fn_df(df_pred.squeeze(1), ds_true_df)
                        loss = reg_loss_weight*reg_loss + df_loss
                        class_loss = None


                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                        'train loss total': loss.item(),
                        'train loss regression': reg_loss.item() if reg_loss is not None else 0.0,
                        'train loss classification': class_loss.item() if class_loss is not None else 0.0,
                        'train loss df': df_loss.item() if df_loss is not None else 0.0,
                        'step': global_step,
                        'epoch': epoch
                    })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                # division_step = (n_train // (10 * batch_size))
                division_step = 30
                if division_step > 0 and global_step % division_step == 0:
                    histograms = {}
                    for tag, value in model.named_parameters():
                        tag = tag.replace('/', '.')
                        if not (torch.isinf(value) | torch.isnan(value)).any():
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                    val_score_cl, val_score_rg, val_score_df = evaluate(model, val_loader, device, amp, 
                                                          use_depth=use_depth, use_mono_depth = use_mono_depth,
                                                          head_mode = head_mode,
                                                          reg_ds_factor=reg_ds_factor)

                    if head_mode == 'both':
                        scheduler.step(val_score_cl - val_score_rg)
                        binary_mask = F.sigmoid(binary_pred.squeeze(1)) > 0.5
                        ds_binary_mask = downsample_torch_mask(binary_mask.float(), reg_ds_factor, "nearest") if reg_ds_factor != 1.0 else binary_mask
                        wandb_mask_pred = masks_pred.squeeze(1) * (F.sigmoid(ds_binary_mask) > 0.5)

                    elif head_mode == 'segmentation':
                        scheduler.step(1 - val_score_cl)
                        binary_mask = F.sigmoid(binary_pred.squeeze(1)) > 0.5
                        # masks_pred does not exist in this case, put a dummy tensor
                        masks_pred = torch.zeros_like(true_masks)
                        wandb_mask_pred = masks_pred.squeeze(1) * (F.sigmoid(binary_pred.squeeze(1))> 0.5)

                    elif head_mode == 'regression':
                        scheduler.step(1 - val_score_rg)
                        wandb_mask_pred = masks_pred.squeeze(1)
                        # binary_mask does not exist in this case, put a dummy tensor
                        binary_mask = torch.zeros_like(true_masks)
                        
                        wandb_df_pred = torch.zeros_like(true_masks)
                        ds_true_df = torch.zeros_like(true_masks)
                    elif head_mode == 'df':
                        scheduler.step(1 - val_score_df)
                        # wandb_df_pred = df_pred.squeeze(1)
                        # if use normalized df, need to denomalize and vis
                        wandb_df_pred = denormalize_df(df_pred,df_neighborhood=10).squeeze(1)
                        # binary_mask does not exist in this case, put a dummy tensor
                        wandb_mask_pred = torch.zeros_like(true_masks)
                        binary_mask = torch.zeros_like(true_masks)

                    elif head_mode == "df_wf":
                        scheduler.step(1 - val_score_df - reg_loss_weight*val_score_rg)
                        wandb_df_pred = denormalize_df(df_pred,df_neighborhood=10).squeeze(1)
                        masks_pred = reverse_log_transform(masks_pred)
                        wandb_mask_pred = masks_pred.squeeze(1) * (wandb_df_pred < 10)
                        binary_mask = torch.zeros_like(true_masks)

                    logging.info(f'Validation Classification Dice score: {val_score_cl}')
                    logging.info(f'Validation Regression mse : {val_score_rg}')
                    logging.info(f'Validation distance field mse : {val_score_df}')

                    # since image could be 4 channels, we need to convert it to 3 channels to get the rgb image
                    wandb_rgb = images[:, :3, :, :]
                    wandb_depth = depth.squeeze(1)
        
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
            torch.save(state_dict, str(dir_checkpoint / f'CP_epoch{epoch}_{use_depth_str}_{reg_weights}.pth'))
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
    parser.add_argument('--use_mono_depth','-umd', action='store_true', default=False, help='Use mono depth image')
    parser.add_argument('--head_mode', type=str, default='segmentation', help='both or segmentation or regression')
    parser.add_argument('--regression_downsample_factor','-rdf', type=float, default=1.0, help='Downsample factor for regression head')
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
    if args.use_depth:
        model = TwoHeadUnet(classes=args.classes,
                            in_channels=4,
                            head_config = head_mode,
                            regression_downsample_factor=args.regression_downsample_factor)
        
    else:
        model = TwoHeadUnet(classes=args.classes,
                            in_channels=3,
                            head_config = head_mode,
                            regression_downsample_factor=args.regression_downsample_factor)
        
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n')
                #  f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
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
            use_mono_depth = args.use_mono_depth,
            reg_loss_weight=args.reg_loss_weight,
            head_mode = head_mode,
            weight_decay=1e-8,
            reg_ds_factor=args.regression_downsample_factor
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
            use_mono_depth = args.use_mono_depth,
            reg_loss_weight=args.reg_loss_weight,
            head_mode = head_mode,
            weight_decay=1e-8,
            reg_ds_factor=args.regression_downsample_factor
        )
