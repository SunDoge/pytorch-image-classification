// Train ResNet18 on imagenette
local ds = import './datasets/cifar.libsonnet';
local models = import './models/cifar_resnet.libsonnet';
local optimizers = import './optimizers.libsonnet';
local trans = import './transforms/cifar.libsonnet';
{
  engine: 'lib.engines.supervised_engine.Engine',
  max_epochs: 10,

  model: models.cifar_resnet20(),
  optimizer: optimizers.SGD(0.1),


  train: {
    transform: trans.supervised.train,
    dataset: ds.cifar10(
      './data',
      train=true,
    ),
    loader: {
      _name: 'flame.pytorch.helpers.create_data_loader',
      dataset: null,
      num_workers: 2,
      batch_size: 128,
    },
  },
  val: {
    transform: trans.supervised.val,
    dataset: ds.cifar10(
      './data',
      train=false,
    ),
    loader: {
      _name: 'flame.pytorch.helpers.create_data_loader',
      dataset: null,
      num_workers: 2,
      batch_size: 128,
    },
  },
}
