import albumentations as A
from albumentations.pytorch import ToTensorV2

def train_transforms():
    train_transforms = A.Compose([A.Normalize(
        mean=(0.49139968, 0.48215841, 0.44653091),
        std=(0.24703223, 0.24348513, 0.26158784),
    ),
    A.PadIfNeeded(40,40),
    A.RandomCrop(32,32),
    A.HorizontalFlip(),
    A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=[0.49139968, 0.48215841, 0.44653091], always_apply=True, p=0.50),
    #A.ShiftScaleRotate(),
    #A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=(0.49139968, 0.48215841, 0.44653091), mask_fill_value=None)
    ])
    return train_transforms

def test_transforms():
    test_transforms = A.Compose([A.Normalize(
        mean=[0.49139968, 0.48215841, 0.44653091],
        std=[0.24703223, 0.24348513, 0.26158784],
    )
    ])
    return test_transforms
