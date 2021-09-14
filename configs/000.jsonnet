// Train ResNet18 on imagenette
local transform = import './transforms/imagenet.libsonnet';

{
  train_transform: transform.supervised.train(),
}
