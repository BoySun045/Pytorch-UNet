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
    elif ext == '.npz':
        return Image.fromarray(np.load(filename)['depth'])
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    # print("mask_dir: ", mask_dir)
    # print("mask_suffix: ", mask_suffix)
    # print("idx: ", idx)
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    # print("mask_file: ", mask_file)
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, depth_dir: str, scale: float = 1.0, mask_suffix: str = ''):
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

        # with Pool() as p:
        #     unique = list(tqdm(
        #         p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
        #         total=len(self.ids)
        #     ))

        # the above loop is too slow when the number of images is large, since I know for each image the mask values are the same and it is [0, 1], I can just hard code it
        unique = [np.array([0, 1]) for _ in self.ids]

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask, is_depth):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        if not is_depth:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img
        
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

    # def preprocess(mask_values, pil_img, scale, is_mask, is_depth):

    #     if is_mask:
    #         # if it is mask, the input is directly a np array with weights value 
    #         # do a resize, normalization and return is enough
    #         mask = np.array(pil_img)
    #         # resize the mask using nearest neighbor
    #         mask = np.array(Image.fromarray(mask).resize((int(mask.shape[1] * scale), int(mask.shape[0] * scale),), resample=Image.NEAREST))
    #         binary_mask = (mask > 0).astype(np.int64)
    #         return binary_mask

    #     else:
    #         w, h = pil_img.size
    #         newW, newH = int(scale * w), int(scale * h)
    #         assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
    #         pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
    #         img = np.asarray(pil_img)

    #         if is_depth:
    #             pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
    #             img = np.asarray(pil_img)
    #             if img.ndim == 2:
    #                 img = img[np.newaxis, ...]
    #             else:
    #                 img = img.transpose((2, 0, 1))

    #             # normalize depth 
    #             img_min = img.min()
    #             img_max = img.max()
    #             img = (img - img_min) / (img_max - img_min)

    #             return img
            
    #         if not is_depth:
    #             pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
    #             img = np.asarray(pil_img)
    #             if img.ndim == 2:
    #                 img = img[np.newaxis, ...]
    #             else:
    #                 img = img.transpose((2, 0, 1))

    #             if (img > 1).any():
    #                 img = img / 255.0

    #             return img
            
    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        if self.depth_dir is not None:
            depth_file = list(self.depth_dir.glob(name + '.*'))
            assert len(depth_file) == 1, f'Either no depth image or multiple depth images found for the ID {name}: {depth_file}'
            depth = load_image(depth_file[0])
            depth = self.preprocess(self.mask_values, depth, self.scale, is_mask=False, is_depth=True)

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False, is_depth=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True, is_depth=False)

        if self.depth_dir is not None:
            return {
                'image': torch.as_tensor(img.copy()).float().contiguous(),
                'mask': torch.as_tensor(mask.copy()).long().contiguous(),
                'depth': torch.as_tensor(depth.copy()).float().contiguous()
            }
        else:
            return {
                'image': torch.as_tensor(img.copy()).float().contiguous(),
                'mask': torch.as_tensor(mask.copy()).long().contiguous()
            }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, depth_dir, scale=1):
        super().__init__(images_dir, mask_dir, depth_dir, scale, mask_suffix='')
