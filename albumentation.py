import albumentations as A
from albumentations.pytorch import ToTensorV2

def train_transforms():
    train_transforms = A.Compose([A.Normalize(
        mean=(0.49139968, 0.48215841, 0.44653091),
        std=(0.24703223, 0.24348513, 0.26158784),
    ),
    A.PadIfNeeded(40),
    A.RandomCrop(32,32),
    A.HorizontalFlip(),
    A.Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=[0.4914*255, 0.4822*255, 0.4471*255], always_apply=True, p=0.50),
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


self.albumentations_transform = Compose([
            PadIfNeeded(40),
            RandomCrop(32,32),
            HorizontalFlip(),
            Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=[0.4914*255, 0.4822*255, 0.4471*255], always_apply=True, p=0.50),
#            CoarseDropout(max_holes=3, max_height=8, max_width=8, min_holes=None, min_height=4, min_width=4, fill_value=[0.4914*255, 0.4822*255, 0.4471*255], mask_fill_value=None, always_apply=False, p=0.7),
            Normalize(mean=[0.4914, 0.4822, 0.4471],std=[0.2469, 0.2433, 0.2615]),
            ToTensorV2()
        ])