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
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

# dir_img = Path('/cluster/project/cvg/boysun/MH3D_train_set_mini/image/')
# dir_mask = Path('/cluster/project/cvg/boysun/MH3D_train_set_mini/mask/')
# dir_depth = Path('/cluster/project/cvg/boysun/MH3D_train_set_mini/depth/')
# # dir_checkpoint = Path('/cluster/project/cvg/boysun/MH3D_train_set_mini/checkpoint_RGBD_miniset/')
# # dir_checkpoint = Path('/cluster/project/cvg/boysun/MH3D_train_set_mini/checkpoint_RGB_miniset/')
# dir_checkpoint = Path('/cluster/project/cvg/boysun/MH3D_train_set_mini/checkpoint_D_miniset/')

# dir_img = Path('/cluster/project/cvg/boysun/MH3D_train_set_mini/image/')
# dir_mask = Path('/cluster/project/cvg/boysun/MH3D_train_set_mini/mask/')
# dir_depth = Path('/cluster/project/cvg/boysun/MH3D_train_set_mini/depth_anything/')
# dir_checkpoint = Path('/cluster/project/cvg/boysun/MH3D_train_set_mini/checkpoint_RGB_D_pred_miniset/')
# dir_checkpoint = Path('/cluster/project/cvg/boysun/MH3D_train_set_mini/checkpoint_RGB_D_pred_miniset/')
# dir_checkpoint = Path('/cluster/project/cvg/boysun/MH3D_train_set_mini/checkpoint_D_pred_miniset/')

# dir_img = Path('/cluster/project/cvg/boysun/MH3D_train_set_middle/image/')
# dir_mask = Path('/cluster/project/cvg/boysun/MH3D_train_set_middle/mask/')
# dir_depth = Path('/cluster/project/cvg/boysun/MH3D_train_set_middle/depth/')
# # dir_checkpoint = Path('/cluster/project/cvg/boysun/MH3D_train_set_mini/checkpoint_RGBD_miniset/')
# dir_checkpoint = Path('/cluster/project/cvg/boysun/MH3D_train_set_middle/checkpoint_RGB_miniset/')

# dir_path = Path("/cluster/project/cvg/boysun/Actmap_v2_mini/")
dir_path = Path("/cluster/project/cvg/boysun/MH3D_train_set_mini/")
dir_img = Path(dir_path / 'image/')
dir_mask = Path(dir_path / 'mask/')
dir_checkpoint = Path(dir_path / 'checkpoints/')


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
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        use_depth: bool = False,
        only_depth: bool = False
):
    # 1. Create dataset
    if use_depth:
        try:
            dataset = CarvanaDataset(dir_img, dir_mask,dir_depth, img_scale)
        except (AssertionError, RuntimeError, IndexError):
            dataset = BasicDataset(dir_img, dir_mask, dir_depth, img_scale)
    else:
        try:
            dataset = CarvanaDataset(dir_img, dir_mask, None, img_scale)
        except (AssertionError, RuntimeError, IndexError):
            dataset = BasicDataset(dir_img, dir_mask, None,img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    print(f"Train size: {n_train}, Validation size: {n_val}")

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=16, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net-v2', resume='allow', anonymous='must')
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
    # optimizer = optim.RMSprop(model.parameters(),
    #                           lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    #use adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    pos_weight = torch.tensor([2.0]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else loss_fn
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch_idx, batch in enumerate(train_loader):

                if not use_depth and not only_depth:  # RGB images
                    images, true_masks = batch['image'], batch['mask']

                    assert images.shape[1] == model.n_channels, \
                        f'Network has been defined with {model.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                
                elif use_depth and not only_depth:  # RGB-D images
                    images, true_masks, depth = batch['image'], batch['mask'], batch['depth']
                    print("image channel plus depth channel: ", images.shape[1] + depth.shape[1])
                    assert images.shape[1] + depth.shape[1] == model.n_channels, \
                        f'Network has been defined with {model.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1] + depth.shape[1]}  channels. Please check that ' \
                        'the images are loaded correctly.'
                    images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    depth = depth.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    images = torch.cat([images, depth], dim=1)
                
                elif only_depth: # Depth images
                    assert batch['depth'].shape[1] == model.n_channels, \
                        f'Network has been defined with {model.n_channels} input channels, ' \
                        f'but loaded images have {batch["depth"].shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'
                        
                    true_masks, depth = batch['mask'], batch['depth']
                    images = depth.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
        
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                # division_step = (n_train // (5 * batch_size))
                division_step = 200
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp, use_depth=use_depth, only_depth=only_depth)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            #convert mask_pred to a binary mask for wandb logging
                            wandb_mask_pred = (F.sigmoid(masks_pred.squeeze(1)) > 0.5).float()
                    
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(wandb_mask_pred[0].cpu())
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass
                
        # if n_train > 10000:
        #     save_every_spoch = 1
        # else:
        #     save_every_spoch = 5
        save_every_spoch = 20
        if save_checkpoint and epoch % save_every_spoch == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
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

    parser.add_argument('--use_depth', action='store_true', default=False, help='Use depth image')
    parser.add_argument('--use_only_depth', action='store_true', default=False, help='Use only depth image')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_channels=4 for RGB-D images
    # n_channels=1 for depth images
    # n_classes is the number of probabilities you want to get per pixel

    # first, check when use_only_depth is enabled, use_depth should also be enabled
    if args.use_only_depth:
        args.use_depth = True
        print('use_depth is forced to be enabled because use_only_depth is enabled')

    if args.use_depth and not args.use_only_depth:
        model = UNet(n_channels=4, n_classes=args.classes, bilinear=args.bilinear) # RGB-D images
    elif args.use_only_depth:
        model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear) # Depth images
    else:
        model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear) # RGB images

    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

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
            only_depth=args.use_only_depth
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
            only_depth=args.use_only_depth
        )
