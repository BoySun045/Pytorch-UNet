import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.data_augmentation import get_transforms

# def load_image(filename):
#     ext = splitext(filename)[1]
#     if ext == '.npy':
#         return Image.fromarray(np.load(filename))
#     elif ext == '.npz':
#         return Image.fromarray(np.load(filename)['depth'])
#     elif ext in ['.pt', '.pth']:
#         return Image.fromarray(torch.load(filename).numpy())
#     else:
#         return Image.open(filename)

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    if ext == '.npz':
        weights = np.load(filename)['weights']
        return weights
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)



class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, depth_dir: str = None, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        if depth_dir is not None:
            print("Using depth iamge in dataset")
            self.depth_dir = Path(depth_dir)
        else:
            self.depth_dir = None

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')

        # data augmentation
        self.transforms = get_transforms()

    def __len__(self):
        return len(self.ids)

    # @staticmethod
    # def preprocess(mask_values, pil_img, scale, is_mask, is_depth):
    #     w, h = pil_img.size
    #     newW, newH = int(scale * w), int(scale * h)
    #     assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
    #     pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
    #     img = np.asarray(pil_img)

    #     if is_mask:
    #         mask = np.zeros((newH, newW), dtype=np.int64)
    #         for i, v in enumerate(mask_values):
    #             if img.ndim == 2:
    #                 mask[img == v] = i
    #             else:
    #                 mask[(img == v).all(-1)] = i

    #         return mask

    #     if not is_depth:
    #         if img.ndim == 2:
    #             img = img[np.newaxis, ...]
    #         else:
    #             img = img.transpose((2, 0, 1))

    #         if (img > 1).any():
    #             img = img / 255.0

    #         return img
        
    #     if is_depth:
    #         if img.ndim == 2:
    #             img = img[np.newaxis, ...]
    #         else:
    #             img = img.transpose((2, 0, 1))

    #         # depth should be with only one channel
    #         if not img.shape[0] == 1:
    #             # make it with only one channel but keep the same dimension (1, H, W)
    #             img = img[0:1, ...]

    #         # normalize depth 
    #         img_min = img.min()
    #         img_max = img.max()
    #         img = (img - img_min) / (img_max - img_min)

    #         return img

    @staticmethod
    def preprocess(pil_img, scale, is_mask, is_depth=False):

        if is_mask:
            # if it is mask, the input is directly a np array with weights value 
            # do a resize, normalization and return is enough
            mask = np.array(pil_img)
            mask_weight_global_max = 2000
            # resize the mask using nearest neighbor
            mask = np.array(Image.fromarray(mask).resize((int(mask.shape[1] * scale), int(mask.shape[0] * scale),), resample=Image.NEAREST))
            mask = np.clip(mask/mask_weight_global_max, 0, 1)
            binary_mask = (mask > 0.0001).astype(np.int64)
            return mask, binary_mask


        else:
            w, h = pil_img.size
            newW, newH = int(scale * w), int(scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
            pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
            img = np.asarray(pil_img)

            if is_depth:
                if img.ndim == 2:
                    img = img[np.newaxis, ...]
                else:
                    img = img.transpose((2, 0, 1))

                # depth should be with only one channel
                if not img.shape[0] == 1:
                    # make it with only one channel but keep the same dimension (1, H, W)
                    img = img[0:1, ...]

                # normalize depth 
                img_min = img.min()
                img_max = img.max()
                img = (img - img_min) / (img_max - img_min)

                return img

            if not is_depth:
                if img.ndim == 2:
                    img = img[np.newaxis, ...]
                else:
                    img = img.transpose((2, 0, 1))

                if (img > 1).any():
                    img = img / 255.0

                return img


    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        # print(mask_file)
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        if self.depth_dir is not None:
            depth_file = list(self.depth_dir.glob(name + '.*'))
            assert len(depth_file) == 1, f'Either no depth image or multiple depth images found for the ID {name}: {depth_file}'
            depth = load_image(depth_file[0])
            depth = self.preprocess(depth, self.scale, is_mask=False, is_depth=True)

        # assert img.size == mask.size, \
        #     f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess( img, self.scale, is_mask=False, is_depth=False)
        mask, binary_mask = self.preprocess(mask, self.scale, is_mask=True, is_depth=False)

        # data augmentation
        # Transpose image and masks to (H, W, C)
        img = np.transpose(img, (1, 2, 0))
        mask = np.transpose(mask, (1, 2, 0)) if mask.ndim == 3 else mask[..., np.newaxis]
        binary_mask = np.transpose(binary_mask, (1, 2, 0)) if binary_mask.ndim == 3 else binary_mask[..., np.newaxis]
        
        sample = {'image': img, 'mask': mask, 'binary_mask': binary_mask}
        # print shape of img mask and binary_mask separately
        # print(f"image shape: {img.shape}")
        # print(f"mask shape: {mask.shape}")
        # print(f"binary_mask shape: {binary_mask.shape}")
        if self.depth_dir is not None:
            sample['depth'] = depth
        if self.transforms:
            augmented = self.transforms(**sample)
            img = augmented['image']
            mask = augmented['mask']
            binary_mask = augmented['binary_mask']
            if self.depth_dir is not None:
                depth = augmented['depth']

        # print(f"image shape after data augmentation: {img.shape}")
        # print(f"mask shape after data augmentation: {mask.shape}")
        # print(f"binary_mask shape after data augmentation: {binary_mask.shape}")

        # Transpose back to (C, H, W)
        # img = np.transpose(img, (2, 0, 1)) 
        mask = np.transpose(mask, (2, 0, 1)).squeeze() if mask.ndim == 3 else mask.squeeze(-1)
        binary_mask = np.transpose(binary_mask, (2, 0, 1)).squeeze() if binary_mask.ndim == 3 else binary_mask.squeeze(-1)


        # print their shape after data augmentation separately
        # print(f"image shape after data augmentation: {img.shape}")
        # print(f"mask shape after data augmentation: {mask.shape}")
        # print(f"binary_mask shape after data augmentation: {binary_mask.shape}")

        assert img.shape[1:] == mask.shape, \
            f'Image and mask {name} should have the same height and width, but are {img.shape[1:]} and {mask.shape}'
        
        if self.depth_dir is not None: 
            return {
                'image': torch.as_tensor(img).float().contiguous(),
                'mask': torch.as_tensor(mask).float().contiguous(),
                'binary_mask': torch.as_tensor(binary_mask).long().contiguous(),
                'depth': torch.as_tensor(depth).float().contiguous()
            }
        
        else:
            return {
                'image': torch.as_tensor(img).float().contiguous(),
                'mask': torch.as_tensor(mask).float().contiguous(),
                'binary_mask': torch.as_tensor(binary_mask).long().contiguous()
            }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, depth_dir, scale=1):
        super().__init__(images_dir, mask_dir, depth_dir, scale, mask_suffix='')
