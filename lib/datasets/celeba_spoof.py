
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


def TrainTransform(size: int):
    bbox_params = A.BboxParams('albumentations')
    return A.Compose([

    ], bbox_params=bbox_params)


def enlarge_bbox(bbox: List[int], scale: float = 1.1, base_size: float = 224.0) -> List[float]:
    """

    output yolo format
    """
    x, y, w, h = map(float, bbox)
    center_x = x + w / 2
    center_y = y + h / 2
    size = max(w, h) * scale
    return [center_x / base_size, center_y / base_size, size / base_size, size / base_size]


class CelebASpoof(Dataset):

    def __init__(
        self,
        root: str,
        file_list: str,
        transform: Callable,
    ) -> None:
        super().__init__()
        with open(file_list, 'rb') as f:
            data = pickle.load(f)

        self.root = Path(root)
        self.samples: List[Sample] = data

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = read_image(str(self.root / sample.image_path))
        label = sample.label
