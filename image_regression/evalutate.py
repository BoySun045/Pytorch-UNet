import torch
import torch.nn as nn
from tqdm import tqdm

@torch.inference_mode()
def evaluate(net, dataloader, device, amp, use_depth=False, 
             use_mono_depth=False):
    
    net.eval()
    num_val_batches = len(dataloader)

    loss_fn_rg = nn.MSELoss()
    loss = 0
    autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'

    with torch.autocast(autocast_device, enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):

            image, mask_true = batch['image'], batch['mask']
            depth = batch['depth'] if not use_mono_depth else batch['mono_depth']
            df = batch['df']

            if use_depth or use_mono_depth:
                image = torch.cat((image, depth), dim=1)

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.float32)

            mask_weight_global_max = 2000.0
            mask_weight_global_min = 1e-6

            mask_max = mask_true.max()
            mask_min = mask_true.min()

            pred = net(image)
            loss += loss_fn_rg(pred, mask_max)

    net.train()
    return loss / num_val_batches