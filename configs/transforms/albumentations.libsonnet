local A(name) = 'albumentations.%s' % name;

{
  Compose(transforms):: {
    _name: A('Compose'),
    transforms: transforms,
  },
  RandomResizedCrop(height, width, scale=[0.08, 1.0]):: {
    _name: A('RandomResizedCrop'),
    height: height,
    width: width,
    scale: scale,
  },
  HorizontalFlip():: {
    _name: A('HorizontalFlip'),
  },
  Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
  ):: {
    _name: A('Normalize'),
    mean: mean,
    std: std,
  },
  ToTensorV2():: {
    _name: A('pytorch.ToTensorV2'),
  },
  SmallestMaxSize(max_size=256):: {
    _name: A('SmallestMaxSize'),
    max_size: max_size,
  },
  CenterCrop(height=224, width=224):: {
    _name: A('CenterCrop'),
    height: height,
    width: width,
  },
  PadIfNeeded(min_height, min_width):: {
    _name: A('PadIfNeeded'),
    min_height: min_height,
    min_width: min_width,
    border_mode: 0,  // cv2.BORDER_CONSTANT
  },
  RandomCrop(height, width):: {
    _name: A('RandomCrop'),
    height: height,
    width: width,
  },
}
