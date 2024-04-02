from torchvision.transforms import Compose, Resize, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, ColorJitter
from torchvision.transforms.functional import to_tensor
import numpy as np
import random
from PIL import Image
import torch

def add_noise(img, mean=0., std=1.):
    """
    Add random Gaussian noise to an image, works with both PIL images and tensors.
    """
    if isinstance(img, torch.Tensor):
        noise = torch.randn(img.size()).to(img.device) * std + mean
        return img + noise
    else:
        # Convert to tensor to add noise
        img_tensor = to_tensor(img)
        noise = torch.randn(img_tensor.size()) * std + mean
        noisy_img = img_tensor + noise  # This is already a tensor, no need to convert again
        return noisy_img

class CustomTransform:
    def __init__(self):
        self.transform = Compose([
            Resize(720),
            RandomCrop(700),
            RandomHorizontalFlip(p=1.0),
            # Further transformations can be added here
        ])
        
        self.color_jitter = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
        
    def __call__(self, image, depth, mask):
        # Apply spatial transformations to all: image, mask, and depth
        seed = np.random.randint(2147483647)  # get a random seed
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.transform(image)
        
        random.seed(seed)
        torch.manual_seed(seed)
        mask = self.transform(mask)

        if depth is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            depth = self.transform(depth)

        # Apply color jitter only to the image
        image = self.color_jitter(image)
        
        # Optionally, add noise to image and depth
        image = add_noise(image, mean=0., std=0.05)
        if depth is not None:
            depth = add_noise(depth, mean=0., std=0.05)
        
        return image, depth, mask
    
