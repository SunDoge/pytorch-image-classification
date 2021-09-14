local A = import './albumentations.libsonnet';

// from https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb#scrollTo=AoliFX5AnBJ0
local mean = [0.4914, 0.4822, 0.4465];
local std = [0.2023, 0.1994, 0.2010];

{
  supervised: {
    train: A.Compose([
      A.PadIfNeeded(32 + 8, 32 + 8),
      A.RandomCrop(32, 32),
      A.HorizontalFlip(),
      A.Normalize(mean, std),
      A.ToTensorV2(),
    ]),
    val: A.Compose([
      A.Normalize(mean, std),
      A.ToTensorV2(),
    ]),
  },
}
