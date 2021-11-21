import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
from tqdm import tqdm

from lib.datasets.celeba_spoof import FaceBbox, Sample
from typing import Optional
import pickle
import dataclasses

_memory = joblib.Memory('./cache', compress=True)


@_memory.cache
def get_images(root: Path):
    images = []
    for path in root.glob('*/*/*'):
        if path.suffix in {'.jpg', '.png'}:
            # index = path.name.split('.')[0]
            # bb_txt = path.parent / f'{index}_BB.txt'
            # if bb_txt.exists():
            #     return path.relative_to(root)
            images.append(path.relative_to(root))

    return images


def read_bb_txt(image_path: Path) -> Optional[FaceBbox]:
    index = image_path.name.split('.')[0]
    bb_txt = image_path.parent / f'{index}_BB.txt'
    content = bb_txt.read_text()
    bb = content.split()[:4]
    if len(bb) < 4:
        return None

    x, y, w, h = list(map(int, content.split()[:4]))
    score = float(content.split()[4])
    return FaceBbox(x, y, w, h, score)


def make_dataset(root_dir: Path, output_file: Path, max_workers: int = 64):
    images = get_images(root_dir)

    samples = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        face_bbox_iter = executor.map(
            lambda image_path: read_bb_txt(root_dir / image_path),
            images,
            chunksize=16,
        )

        for image_path, face_bbox in zip(tqdm(images, ascii=True), face_bbox_iter):
            if face_bbox is None:
                continue

            sample = Sample(
                image_path,
                face_bbox.to_xywh(),
                1 if 'live' in str(image_path) else 0
            )
            samples.append(dataclasses.asdict(sample))

    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(output_file, 'wb') as f:
        pickle.dump(samples, f)

    return samples


def main():
    make_dataset(
        Path('/mnt/cephfs/dataset/FAS/CelebA_Spoof/CelebA_Spoof/Data/train'),
        Path('data/celeba_spoof/train_list.pkl')
    )
    make_dataset(
        Path('/mnt/cephfs/dataset/FAS/CelebA_Spoof/CelebA_Spoof/Data/test'),
        Path('data/celeba_spoof/test_list.pkl')
    )


if __name__ == '__main__':
    main()
