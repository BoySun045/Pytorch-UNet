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
from utils.data_augmentation import get_transforms, get_static_transforms, get_appearance_transforms
from dataset.hm3d_gt import load_image, log_transform_mask, min_max_scale, compute_df, compute_wf, label_wf
import cv2

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, depth_dir: str = None,
                  scale: float = 1.0, 
                  gen_mono_depth: bool = False,
                  mask_suffix: str = '', data_augmentation=True, log_transform=True):
        
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
        self.transforms = get_transforms() if data_augmentation else get_static_transforms()
        self.img_app_transforms = get_appearance_transforms() if data_augmentation else None
        print("if do data augmentation: ", data_augmentation)

        # log loss transform
        self.log_transform = log_transform

        # set mono depth path
        self.mono_depth = gen_mono_depth
            # mono depth dir is the same path as depth_dir but with name mono_depth instead of depth
        self.mono_depth_dir = self.depth_dir.parent / 'mono_depth' if self.mono_depth else None

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask, is_depth=False, log_transform=True):

        if is_mask:
            # if it is mask, the input is directly a np array with weights value 
            # do a resize, normalization and return is enough
            mask = np.array(pil_img)
            mask = np.array(Image.fromarray(mask).resize((int(mask.shape[1] * scale), int(mask.shape[0] * scale),), resample=Image.NEAREST))

            # global min_max
            mask_weight_global_max = 3000.0
            mask_weight_global_min = 0
            
            # local min_max
            # mask_weight_global_max = mask.max()
            # mask_weight_global_min = mask[mask>0].min()
            #print("weighted mask max min", mask_weight_global_max, mask_weight_global_min)

            # handling nan sitation:
            if abs(mask_weight_global_max-mask_weight_global_min) < 1e-3 or mask_weight_global_max < mask_weight_global_min:
                mask = np.ones_like(mask)
                binary_mask = (mask > 0.0001).astype(np.int64)
                return mask, binary_mask

            if log_transform:
                mask = np.clip(mask, mask_weight_global_min, mask_weight_global_max)
                mask_log = log_transform_mask(mask)   # log transform move to the loss calculation part
                # mask_log_max = np.log1p(mask_weight_global_max)
                # mask_log_min = np.log1p(mask_weight_global_min)
                # mask_log = min_max_scale(mask_log, mask_log_min, mask_log_max)
                # print("dataloader mask min max", mask.min(), mask.max())
                mask = mask_log

                # mask = np.clip(mask, 0, 1)
                # mask = mask_log
                # print("mask min max: ", mask.min(), mask.max())
            
            else:
                mask = np.clip(mask, mask_weight_global_min, mask_weight_global_max)
                mask = min_max_scale(mask, mask_weight_global_min, mask_weight_global_max)
                mask = np.clip(mask, 0, 1)

            binary_mask = (mask > 0.0001).astype(np.int64)
            return mask, binary_mask


        else:

            if is_depth:

                w = pil_img.shape[1]
                h = pil_img.shape[0]
                newW, newH = int(scale * w), int(scale * h)
                assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
                pil_img = cv2.resize(pil_img, (newW, newH), interpolation=cv2.INTER_CUBIC)
                img = np.asarray(pil_img).astype(np.float32)
            
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

                w, h = pil_img.size
                newW, newH = int(scale * w), int(scale * h)
                assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
                pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
                img = np.asarray(pil_img)

                if img.ndim == 2:
                    img = img[np.newaxis, ...]
                else:
                    img = img.transpose((2, 0, 1))

                if (img > 1).any():
                    img = img / 255.0

                return img
    
    @staticmethod
    def get_df(mask, depth, scale, df_neighbourhood=10):
        
        mask = np.array(mask)
        mask = np.array(Image.fromarray(mask).resize((int(mask.shape[1] * scale), int(mask.shape[0] * scale),), resample=Image.NEAREST))
        binary_mask = (mask > 0.0001).astype(np.float32)

        w = depth.shape[1]
        h = depth.shape[0]
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        depth = cv2.resize(depth, (newW, newH), interpolation=cv2.INTER_CUBIC)
        depth = np.asarray(depth).astype(np.float32)
        
        assert mask.shape == depth.shape, f'Mask and depth should have the same size, but are {mask.shape} and {depth.shape}'

        df = compute_df(binary_mask, depth, df_neighbourhood)
        return df
    
    @staticmethod
    def get_wf(mask, distance_field, scale, wf_neighbourhood=1.0):
        mask = np.array(mask)
        mask = np.array(Image.fromarray(mask).resize((int(mask.shape[1] * scale), int(mask.shape[0] * scale),), resample=Image.NEAREST))
        wf = compute_wf(mask, distance_field, wf_neighbourhood)
        return wf


    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        # print(mask_file)
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        # if self.depth_dir is not None:  # TODO: fix this, we always need depth image
        depth_file = list(self.depth_dir.glob(name + '.*'))
        assert len(depth_file) == 1, f'Either no depth image or multiple depth images found for the ID {name}: {depth_file}'
        depth = load_image(depth_file[0], load_depth=True)
        df = self.get_df(mask, depth, self.scale)
        depth = self.preprocess(depth, self.scale, is_mask=False, is_depth=True)

        # run mono-depth 
        if self.mono_depth:
            mono_depth_file = list(self.mono_depth_dir.glob(name + '.*'))
            assert len(mono_depth_file) == 1, f'Either no mono depth image or multiple mono depth images found for the ID {name}: {mono_depth_file}'
            mono_depth = load_image(mono_depth_file[0], load_depth=True)
            mono_depth = self.preprocess(mono_depth, self.scale, is_mask=False, is_depth=True)

        img = self.preprocess( img, self.scale, is_mask=False, is_depth=False)
        mask, binary_mask = self.preprocess(mask, self.scale, is_mask=True, is_depth=False, log_transform=self.log_transform)
        
        # get the weight field
        # print("mask min max before get_wf", mask.min(), mask.max())
        mask = self.get_wf(mask, df, 1.0, 10.0) # scale does not need to be changed here since previous preprocess already did resize

        # get labeled mask
        label_mask = label_wf(mask, num_bins=30, end=8.5, start=0, exp_max=20)
            # mask out the invalid value, set its label to 0
        label_mask[np.where(df > 10)] = 0
        label_mask[label_mask >19] = 19
        # print("label mask unique", np.unique(label_mask))

        # data augmentation
        if self.transforms:
            img = np.transpose(img, (1, 2, 0))
            mask = np.transpose(mask, (1, 2, 0)) if mask.ndim == 3 else mask[..., np.newaxis]
            binary_mask = np.transpose(binary_mask, (1, 2, 0)) if binary_mask.ndim == 3 else binary_mask[..., np.newaxis]
            label_mask = np.transpose(label_mask, (1, 2, 0)) if label_mask.ndim == 3 else label_mask[..., np.newaxis]

            sample = {'image': img, 'mask': mask, 'binary_mask': binary_mask, 'label_mask': label_mask}

            if self.depth_dir is not None:
                depth = np.transpose(depth, (1, 2, 0)) if depth.ndim == 3 else depth[..., np.newaxis]
                sample['depth'] = depth
                df = np.transpose(df, (1, 2, 0)) if df.ndim == 3 else df[..., np.newaxis]
                sample['df'] = df
            
            if self.mono_depth:
                mono_depth = np.transpose(mono_depth, (1, 2, 0)) if mono_depth.ndim == 3 else mono_depth[..., np.newaxis]
                sample['mono_depth'] = mono_depth

            augmented = self.transforms(**sample)
            img = augmented['image']
            mask = augmented['mask']
            binary_mask = augmented['binary_mask']
            label_mask = augmented['label_mask']

            if self.depth_dir is not None:
                depth = augmented['depth']
                df = augmented['df']
                df = np.transpose(df, (2, 0, 1)).squeeze() if df.ndim == 3 else df.squeeze(-1)

            if self.mono_depth:
                mono_depth = augmented['mono_depth']

            if self.img_app_transforms is not None:
                # do appearance transform only for img
                    # Convert img back to numpy array for appearance transforms
                img = img.cpu().numpy().transpose(1, 2, 0)  # Ensure it is a numpy array in HWC format
                img = img.astype(np.float32)
                sample_ = {'image': img}
                img = self.img_app_transforms(**sample_)['image']

            # Transpose back to (C, H, W)
            mask = np.transpose(mask, (2, 0, 1)).squeeze() if mask.ndim == 3 else mask.squeeze(-1)
            binary_mask = np.transpose(binary_mask, (2, 0, 1)).squeeze() if binary_mask.ndim == 3 else binary_mask.squeeze(-1)
            label_mask = np.transpose(label_mask, (2, 0, 1)).squeeze() if label_mask.ndim == 3 else label_mask.squeeze(-1)

        assert img.shape[1:] == mask.shape, \
            f'Image and mask {name} should have the same height and width, but are {img.shape[1:]} and {mask.shape}'

        if self.mono_depth: 
            return {
                'image': torch.as_tensor(img).float().contiguous(),
                'mask': torch.as_tensor(mask).float().contiguous(),
                'binary_mask': torch.as_tensor(binary_mask).long().contiguous(),
                'depth': torch.as_tensor(depth).float().contiguous(),
                'df': torch.as_tensor(df).float().contiguous(),
                'label_mask': torch.as_tensor(label_mask).long().contiguous(),
                'mono_depth': torch.as_tensor(mono_depth).float().contiguous()
            }
        
        else:
            return {
                'image': torch.as_tensor(img).float().contiguous(),
                'mask': torch.as_tensor(mask).float().contiguous(),
                'binary_mask': torch.as_tensor(binary_mask).long().contiguous(),
                'depth': torch.as_tensor(depth).float().contiguous(),
                'df': torch.as_tensor(df).float().contiguous(),
                'label_mask': torch.as_tensor(label_mask).long().contiguous()
            }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, depth_dir, 
                scale=1, 
                gen_mono_depth = False,
                data_augmentation=True, log_transform=True):
        
        super().__init__(images_dir, mask_dir, depth_dir, 
                        scale, 
                        gen_mono_depth = gen_mono_depth,
                        mask_suffix='', data_augmentation=data_augmentation, log_transform=log_transform)
