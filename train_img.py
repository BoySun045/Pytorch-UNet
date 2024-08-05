import torch.optim as optim
from image_regression.model import RGBDResNetRegression
from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from image_regression.dataloader import BasicDataset
import datetime 
import argparse
import logging
import torch
import wandb
from tqdm import tqdm
from image_regression.evalutate import evaluate

dir_path = Path("/mnt/boysunSSD/Actmap_v2_mini")
# dir_path = Path("/cluster/project/cvg/boysun/Actmap_v3")  # actmap_v3 is the one after data balancing cleaning
# dir_path = Path("/cluster/project/cvg/boysun/Actmap_v2_mini")
# dir_path = Path("/cluster/project/cvg/boysun/one_image_dataset_3")
# dir_path = Path("/mnt/boysunSSD//one_image_dataset_3")
dir_img = Path(dir_path / 'image/')
dir_mask = Path(dir_path / 'weighted_mask/')
dir_checkpoint = Path(dir_path / 'img_reg_checkpoints' / datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
dir_debug = Path(dir_path / 'debug/')
dir_depth = Path(dir_path / 'depth/')

def get_args():
    parser = argparse.ArgumentParser(description='Train the image regressor on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=2e-6,
                        help='Learning rate', dest='lr')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=0.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--use_depth','-ud', action='store_true', default=False, help='Use depth image')
    parser.add_argument('--use_mono_depth','-umd', action='store_true', default=False, help='Use mono depth image')
    return parser.parse_args()

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
        dataset_portion: float = 1.0,
        lr_decay: bool = True,
        log_transform = True,
):
    # 1. Create dataset
    data_augmentation = False
    log_transform = log_transform

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
    loader_args = dict(batch_size=batch_size, num_workers=16, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='Resnet', resume='allow', anonymous='must')
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
             lr_decay = lr_decay,
             log_transform = log_transform,
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
    # loss_fn_rg = nn.MSELoss()
    loss_fn_rg = nn.L1Loss()
    global_step = 0 
    
    # 6. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for _, batch in enumerate(train_loader):

                true_masks = batch['mask']
                images = batch['image']
                depth = batch['depth'] if not use_mono_depth else batch['mono_depth']
                true_df = batch['df']

                assert images.shape[1] + depth.shape[1] == 4, \
                    f'Network has been defined with {4} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                depth = depth.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

                if use_depth:
                    images = torch.cat([images, depth], dim=1) if use_depth else images
                
                # mask the valid pixels from true_masks
                valid_mask = (true_df<10).float()
                print("true mask shape: ", true_masks.shape)
                # get true mask max and min per batch
                batch_size = true_masks.shape[0]
                true_max = true_masks.view(batch_size, -1).max(dim=1).values
                true_min = true_masks.view(batch_size, -1).max(dim=1).values
                print(f"true_max: {true_max}, true_min: {true_min}")
                true_max = true_max.to(device=device, dtype=torch.float32)
                true_min = true_min.to(device=device, dtype=torch.float32)
                # true_min_max = torch.cat([true_min, true_max], dim=0).to(device=device, dtype=torch.float32)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                
                    pred = model(images)
                    diff = torch.abs(pred.squeeze() - true_max.squeeze())
                    print("diff: ", diff)
                    print("diff shape: ", diff.shape)
                    print("diff mean: ", diff.mean())
                
                    # loss = loss_fn_rg(pred, true_min_max)
                    print(f"pred: {pred}, true_max: {true_max}")
                    loss = loss_fn_rg(pred, true_max)
                    print("loss: ", loss)

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
                        'step': global_step,
                        'epoch': epoch
                    })
                pbar.set_postfix(**{'loss (batch)': loss.item()})
            
                # Evaluation round
                # division_step = (n_train // (10 * batch_size))
                division_step = 1000
                if division_step > 0 and global_step % division_step == 0:

                    val_score_cl = evaluate(model, val_loader, device, amp, 
                                            use_depth=use_depth, use_mono_depth=use_mono_depth)
                    
                    scheduler.step(1 - val_score_cl)

                    experiment.log({
                        'learning rate': optimizer.param_groups[0]['lr'],
                        'validation avg score classification': val_score_cl,
                        'step': global_step,
                        'epoch': epoch
                    })
                                
                    
        if save_checkpoint and epoch % 1 == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            use_depth_str = 'depth' if use_depth else 'no_depth'
            torch.save(state_dict, str(dir_checkpoint / f'CP_epoch{epoch}_{use_depth_str}_.pth'))
            logging.info(f'Checkpoint {epoch} saved!')


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if args.use_depth or args.use_mono_depth:
        model = RGBDResNetRegression(base_model ='resnet50',
                                      in_channels=4,
                                      kernel_size=5)
    
    else:
        model = RGBDResNetRegression(base_model ='resnet50',
                                      in_channels=3,
                                      kernel_size=5)

    model = model.to(device)


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
            weight_decay=1e-8,
            save_checkpoint = False,
        )

    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
