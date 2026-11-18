import torch
from monai.transforms import (
    CastToType,
    Compose,
    RandFlip,
    RandRotate,
    RandScaleIntensity,
    RandZoom,
)


train_transforms_pre = Compose(
    [
        RandRotate(prob=0.3, range_x=0.2),
        RandRotate(prob=0.3, range_y=0.2),
        RandRotate(prob=0.3, range_z=0.2),
        CastToType(dtype=torch.float32),
    ]
)

valid_transforms_pre = Compose(
    [
        CastToType(dtype=torch.float32),
    ]
)

train_transforms_pre_severe = Compose(
    [
        RandRotate(prob=0.5, range_x=0.2),
        RandRotate(prob=0.5, range_y=0.2),
        RandRotate(prob=0.5, range_z=0.2),
        RandFlip(prob=0.3, spatial_axis=0),
        RandZoom(prob=0.3, min_zoom=0.95, max_zoom=1.05, mode='trilinear'),
        RandScaleIntensity(factors=0.05, prob=0.4),
        CastToType(dtype=torch.float32),
    ]
)
