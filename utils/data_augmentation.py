import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms():
    transforms = A.Compose([
        A.RandomCrop(width=320, height=320),
        A.HorizontalFlip(p=0.5),
        ToTensorV2()
    ], additional_targets={'mask': 'mask', 'binary_mask': 'mask', 'depth': 'image'})

    # set is_check_shape=False to avoid assertion error
    return transforms

def get_static_transforms():
    transforms = A.Compose([
        A.CenterCrop(width=320, height=320),
        ToTensorV2()
    ], additional_targets={'mask': 'mask', 'binary_mask': 'mask', 'depth': 'image'})

    # set is_check_shape=False to avoid assertion error
    return transforms