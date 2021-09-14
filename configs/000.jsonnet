// Train ResNet18 on imagenette
local transform = import './transforms/imagenet.libsonnet';

{
  engine: 'lib.engines.supervised_engine.Engine',
}
