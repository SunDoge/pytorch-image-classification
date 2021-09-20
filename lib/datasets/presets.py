import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class ResslCifarTransform:

    def __init__(
        self,
    ) -> None:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        normalize = A.Normalize(mean=mean, std=std)

        self.strong_trasnform = A.Compose([
            A.RandomResizedCrop(32, 32, scale=(0.2, 1.0)),
            A.HorizontalFlip(),
            A.ColorJitter(
                0.4, 0.4, 0.4, 0.1, p=0.8,
            ),
            A.ToGray(p=0.2),
            A.GaussianBlur(p=0.5),
            normalize,
            ToTensorV2(),
        ])

        self.weak_transform = A.Compose([
            A.RandomResizedCrop(32, 32, scale=(0.2, 1.0)),
            A.HorizontalFlip(),
            normalize,
            ToTensorV2(),
        ])

    def __call__(self, image: np.ndarray):
        strong_view = self.strong_trasnform(image=image)['image']
        weak_view = self.weak_transform(image=image)['image']
        return [strong_view, weak_view]
