import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms():
    transforms = A.Compose([
        A.RandomCrop(width=224, height=224),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.0, rotate_limit=45, p=0.5),
        ToTensorV2()
    ], additional_targets={'mask': 'mask', 'binary_mask': 'mask', 
                           'depth': 'image', 'mono_depth': 'image', 'df': 'mask'})

    return transforms

def get_appearance_transforms():
    transforms = A.Compose([
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
        ToTensorV2()
    ])
    return transforms

def get_static_transforms():
    transforms = A.Compose([
        A.CenterCrop(width=224, height=224),
        ToTensorV2()
    ], additional_targets={'mask': 'mask', 'binary_mask': 'mask', 
                           'depth': 'image', 'mono_depth': 'image', 'df': 'mask'})

    # set is_check_shape=False to avoid assertion error
    return transforms