local A = import './albumentations.libsonnet';

{
  supervised: {

    train(size=224):: A.Compose([
      A.RandomResizedCrop(size, size),
      A.HorizontalFlip(),
      A.Normalize(),
      A.ToTensorV2(),
    ]),
    val(crop=224):: A.Compose([
      A.SmallestMaxSize(crop + 32),
      A.CenterCrop(crop),
      A.Normalize(),
      A.ToTensorV2(),
    ]),
  },
}
