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
    def __init__(self, images_dir: str, mask_dir: str, depth_dir: str, scale: float = 1.0, mask_suffix: str = '', weight_global_max: float = 2000.0):
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

        assert weight_global_max > 0, 'Global max value for mask normalization must be greater than 0'
        self.weight_global_max = weight_global_max

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask, is_depth, mask_weight_global_max = None):



        if is_mask:
            # if it is mask, the input is directly a np array with weights value 
            # do a resize, normalization and return is enough
            mask = np.array(pil_img)
            # resize the mask using nearest neighbor
            mask = np.array(Image.fromarray(mask).resize((int(mask.shape[1] * scale), int(mask.shape[0] * scale),), resample=Image.NEAREST))
            mask = np.clip(mask/mask_weight_global_max, 0, 1)
            return mask

        else:
            w, h = pil_img.size
            newW, newH = int(scale * w), int(scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
            pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
            img = np.asarray(pil_img)

            if is_depth:
                pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
                img = np.asarray(pil_img)
                if img.ndim == 2:
                    img = img[np.newaxis, ...]
                else:
                    img = img.transpose((2, 0, 1))

                # normalize depth 
                img_min = img.min()
                img_max = img.max()
                img = (img - img_min) / (img_max - img_min)

                return img
            
            if not is_depth:
                pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
                img = np.asarray(pil_img)
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
        img_file = list(self.images_dir.glob(name + '.*'))
        if self.depth_dir is not None:
            depth_file = list(self.depth_dir.glob(name + '.*'))
            assert len(depth_file) == 1, f'Either no depth image or multiple depth images found for the ID {name}: {depth_file}'
            depth = load_image(depth_file[0])
            depth = self.preprocess(depth, self.scale, is_mask=False, is_depth=True)

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        # assert img.size == mask.size, \
        #     f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False, is_depth=False)
        mask = self.preprocess(mask, self.scale, is_mask=True, is_depth=False, mask_weight_global_max=self.weight_global_max)

        assert img.shape[-2:] == mask.shape[-2:], f'Image and mask {name} should be the same size, but are {img.shape} and {mask.shape}'

        if self.depth_dir is not None:
            return {
                'image': torch.as_tensor(img.copy()).float().contiguous(),
                'mask': torch.as_tensor(mask.copy()).float().contiguous(),
                'depth': torch.as_tensor(depth.copy()).float().contiguous()
            }
        else:
            return {
                'image': torch.as_tensor(img.copy()).float().contiguous(),
                'mask': torch.as_tensor(mask.copy()).float().contiguous()
            }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, depth_dir, scale=1):
        super().__init__(images_dir, mask_dir, depth_dir, scale, mask_suffix='')
