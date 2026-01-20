import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms():
    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5, crop_border=True),
        A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.95, 1.05)),
        ToTensorV2()
    ], additional_targets={'mask': 'mask', 'binary_mask': 'mask', 'label_mask': 'mask',
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
    ], additional_targets={'mask': 'mask', 'binary_mask': 'mask', 'label_mask': 'mask',
                           'depth': 'image', 'mono_depth': 'image', 'df': 'mask'})

    # set is_check_shape=False to avoid assertion error
    return transforms