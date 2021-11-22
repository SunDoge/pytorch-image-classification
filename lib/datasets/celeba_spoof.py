
from pathlib import Path
from typing import Callable, List, Optional
from numpy.core.defchararray import center
from torch.utils.data import Dataset
from dataclasses import dataclass
import pickle
from PIL import Image
import numpy as np
import albumentations as A
import math
from albumentations import bbox_crop
import math
from albumentations.pytorch import ToTensorV2
import logging

_logger = logging.getLogger(__name__)


def read_image(path: str):
    img = Image.open(path)
    img = img.convert('RGB')
    return np.array(img)


@dataclass
class FaceBbox:
    """
    coco format
    """
    x_min: int
    y_min: int
    width: int
    height: int
    score: float

    def to_xyxy(self):
        """
        the x, y, w, h are pixel in 224
        """
        x1 = self.x_min / 224.0
        y1 = self.y_min / 224.0
        w = self.width / 224.0
        h = self.height / 224.0

        x2 = x1 + w
        y2 = y1 + h

        return [x1, y1, x2, y2]

    def to_xywh(self):
        return [self.x_min, self.y_min, self.width, self.height]


@dataclass
class Sample:
    image_path: str
    bbox: List[int]
    label: int  # 0 spoof, 1 live


def TrainTransform(size: int = 224):
    return A.Compose([
        A.Rotate(limit=30),
        A.RandomResizedCrop(size, size),
        A.ColorJitter(),
        A.Normalize(),
        ToTensorV2()
    ])


def ValTransform(size: int = 224):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(),
        ToTensorV2()
    ])


def enlarge_bbox(bbox: List[int], scale: float = 1.1, base_size: float = 224.0) -> List[int]:
    """

    output yolo format
    """
    x, y, w, h = map(float, bbox)
    center_x = x + w / 2
    center_y = y + h / 2
    size = max(w, h) * scale
    # return [center_x / base_size, center_y / base_size, size / base_size, size / base_size]
    half = size / 2
    x1 = max(center_x - half, 0)
    y1 = max(center_y - half, 0)
    x2 = center_x + half
    y2 = center_y + half
    bbox1 = list(map(round, [x1, y1, x2, y2]))
    return bbox1


class CelebASpoof(Dataset):

    def __init__(
        self,
        root: str,
        file_list: str,
        transform: Callable,
    ) -> None:
        super().__init__()
        with open(file_list, 'rb') as f:
            _logger.info('load file_list from %s', file_list)
            data = pickle.load(f)

        self.root = Path(root)
        self.samples: List[Sample] = [Sample(**d) for d in data]
        self.transform = transform

        _logger.info('length: %s', len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        # print('index', index)
        sample = self.samples[index]
        image = read_image(str(self.root / sample.image_path))
        label = sample.label
        bbox = enlarge_bbox(sample.bbox)
        x1, y1, x2, y2 = bbox
        
        # [h, w, c]
        image1 = image[x1:x2, y1:y2]
        if any([x == 0 for x in image1.shape]):
            print(sample, image.shape, bbox)

        # print(sample, image1.shape, bbox)

        image2 = self.transform(image=image1)['image']

        return image2, label


if __name__ == '__main__':
    ds = CelebASpoof(
        '/mnt/cephfs/dataset/FAS/CelebA_Spoof/CelebA_Spoof/Data/train',
        'data/celeba_spoof/train_list.pkl',
        TrainTransform(),
    )
    image, label = ds[1]
    print(image.shape)
    print(label)

    for index, (image, label) in enumerate(ds):
        print(index, image.shape, label)
