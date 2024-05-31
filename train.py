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
from evaluate import evaluate
from unet import UNet, UnetResnet, TwoHeadUnet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from utils.regression_loss import mse_loss, weighted_mse_loss, mae_loss, weighted_huber_loss
from torchvision.utils import save_image

# dir_img = Path('/cluster/project/cvg/boysun/MH3D_train_set_mini/image/')
# dir_mask = Path('/cluster/project/cvg/boysun/MH3D_train_set_mini/mask/')
# dir_depth = Path('/cluster/project/cvg/boysun/MH3D_train_set_mini/depth/')
# dir_checkpoint = Path('/cluster/project/cvg/boysun/MH3D_train_set_mini/')
# dir_debug = Path('/cluster/project/cvg/boysun/MH3D_train_set_mini/debug/')

# dir_path = Path("/media/boysun/Extreme Pro/Actmap_v2_mini")
dir_path = Path("/cluster/project/cvg/boysun/Actmap_v2")
# dir_path = Path("/media/boysun/Extreme Pro/one_image_dataset_3")
dir_img = Path(dir_path / 'image/')
dir_mask = Path(dir_path / 'weighted_mask/')
dir_checkpoint = Path(dir_path / 'checkpoints/')
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
        momentum: float = 0.5,
        gradient_clipping: float = 1.0,
        use_depth: bool = False,
        reg_loss_weight: float = 1.0,
        head_mode: str = 'segmentation'
):
    
    # 1. Create dataset
    data_augmentation = True
    if use_depth:
        try:
            dataset = CarvanaDataset(dir_img, dir_mask,dir_depth, img_scale, data_augmentation=data_augmentation)
        except (AssertionError, RuntimeError, IndexError):
            dataset = BasicDataset(dir_img, dir_mask, dir_depth, img_scale, data_augmentation=data_augmentation)
    else:
        try:
            dataset = CarvanaDataset(dir_img, dir_mask, None, img_scale, data_augmentation=data_augmentation)
        except (AssertionError, RuntimeError, IndexError):
            dataset = BasicDataset(dir_img, dir_mask, None,img_scale, data_augmentation=data_augmentation)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    print(f"Train size: {n_train}, Validation size: {n_val}")

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=64, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net-resnet', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5, min_lr=5e-7)  # goal: maximize score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # loss_fn_rg = weighted_mse_loss
    loss_fn_rg = weighted_huber_loss
    pos_weight = torch.tensor([2.0]).to(device)
    loss_fn_cl = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    global_step = 0 
    class_loss_weight = 1.0
    reg_loss_weight = reg_loss_weight       
    # weight of regression loss really matters, 5.0 is a tested good one, if it's higher, e.g., 10.0, cls result becomes worse

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch_idx, batch in enumerate(train_loader):

                true_masks, true_binary_masks = batch['mask'], batch['binary_mask']

                if not use_depth:
                    images = batch['image']
                    # assert images.shape[1] == model.n_channels, \
                    assert images.shape[1] == 3, \
                        f'Network has been defined with {3} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                else: 
                    images, depth = batch['image'],batch['depth']

                    # assert images.shape[1] + depth.shape[1] == model.n_channels, \
                    assert images.shape[1] + depth.shape[1] == 4, \
                        f'Network has been defined with {4} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    depth = depth.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    images = torch.cat([images, depth], dim=1)

                true_masks = true_masks.to(device=device, dtype=torch.float32)
                true_binary_masks = true_binary_masks.to(device=device, dtype=torch.float32)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):

                    if head_mode == 'both':
                        binary_pred, masks_pred = model(images)
                        reg_loss = loss_fn_rg(masks_pred.squeeze(1), true_masks.float(), true_binary_masks.float(), increase_factor=8.0, avg_using_binary_mask=True)
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
                        reg_loss = loss_fn_rg(masks_pred.squeeze(1), true_masks.float(), true_binary_masks.float(), increase_factor=8.0, avg_using_binary_mask=False)
                        loss = reg_loss

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
                        'step': global_step,
                        'epoch': epoch
                    })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                division_step = 100
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score_cl, val_score_rg = evaluate(model, val_loader, device, amp, use_depth=use_depth, head_mode = head_mode)

                        if head_mode == 'both':
                            scheduler.step(val_score_cl - val_score_rg)
                            binary_mask = F.sigmoid(binary_pred.squeeze(1)) > 0.5
                            wandb_mask_pred = masks_pred.squeeze(1) * (F.sigmoid(binary_pred.squeeze(1))> 0.5)

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

                        logging.info(f'Validation Classification Dice score: {val_score_cl}')
                        logging.info(f'Validation Regression mse : {val_score_rg}')

                        # since image could be 4 channels, we need to convert it to 3 channels to get the rgb image
                        wandb_rgb = images[:, :3, :, :]
                        if use_depth:
                            # depth is the last channel
                            wandb_depth = images[:, -1, :, :]
                            try:      
                                experiment.log({
                                    'learning rate': optimizer.param_groups[0]['lr'],
                                    'validation avg score classification': val_score_cl,
                                    'validation avg score regression': val_score_rg,
                                    'images': wandb.Image(wandb_rgb[0].cpu()),
                                    'depth': wandb.Image(wandb_depth[0].cpu()),
                                    'masks': {
                                        'true': wandb.Image(true_masks[0].float().cpu()),
                                        'true_binary': wandb.Image(true_binary_masks[0].float().cpu()),
                                        'pred': wandb.Image(wandb_mask_pred[0].float().cpu()),
                                        'pred_binary': wandb.Image(binary_mask[0].float().cpu())
                                    },
                                    'step': global_step,
                                    'epoch': epoch,
                                    **histograms
                                })
                            except:
                                pass
                        else:
                            try:
                                experiment.log({
                                    'learning rate': optimizer.param_groups[0]['lr'],
                                    'validation avg score classification': val_score_cl,
                                    'validation avg score regression': val_score_rg,
                                    'images': wandb.Image(wandb_rgb[0].cpu()),
                                    'masks': {
                                        'true': wandb.Image(true_masks[0].float().cpu()),
                                        'true_binary': wandb.Image(true_binary_masks[0].float().cpu()),
                                        'pred': wandb.Image(wandb_mask_pred[0].float().cpu()),
                                        'pred_binary': wandb.Image(binary_mask[0].float().cpu())
                                    },
                                    'step': global_step,
                                    'epoch': epoch,
                                    **histograms
                                })
                            except:
                                pass

        if save_checkpoint and epoch % 100 == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            # state_dict['mask_values'] = dataset.mask_values
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
    parser.add_argument('--head_mode', type=str, default='segmentation', help='both or segmentation or regression')
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
        # model = UNet(n_channels=4, n_classes=args.classes, bilinear=args.bilinear)
        # model = UnetResnet(n_classes=args.classes)
        model = TwoHeadUnet(classes=args.classes,
                            in_channels=4,
                            head_config = head_mode)
    else:
        # model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
        # model = UnetResnet(n_classes=args.classes)
        model = TwoHeadUnet(classes=args.classes,
                            in_channels=3,
                            head_config = head_mode)
        
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
            reg_loss_weight=args.reg_loss_weight,
            head_mode = head_mode,
            weight_decay=1e-6
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
            reg_loss_weight=args.reg_loss_weight,
            head_mode = head_mode,
            weight_decay=1e-6
        )
