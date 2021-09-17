// Train ResNet20 on CIFAR10
local ds = import './datasets/cifar.libsonnet';
local models = import './models/cifar_resnet.libsonnet';
local optimizers = import './optimizers.libsonnet';
local schedulers = import './schedulers.libsonnet';
local trans = import './transforms/cifar.libsonnet';

{
  local this = self,
  // main_worker: 'lib.engines.supervised_engine.Engine',
  _name: 'lib.engines.supervised_engine.main_worker',
  args: '$args',
  max_epochs: 200,
  print_freq: 10,

  model_config: models.cifar_resnet20(self.train_config.dataset.num_classes),
  optimizer_config: optimizers.SGD(0.1),
  scheduler_config: schedulers.MultiStepLR,

  train_config: {
    transform: trans.supervised.train,
    dataset: ds.cifar10(
      './data',
      train=true,
    ),
    loader: {
      _name: 'flame.pytorch.helpers.create_data_loader',
      dataset: '$dataset',
      num_workers: 2,
      batch_size: 128,
      // persistent_workers: false,
    },
  },
  val_config: {
    transform: trans.supervised.val,
    dataset: ds.cifar10(
      './data',
      train=false,
    ),
    loader: {
      _name: 'flame.pytorch.helpers.create_data_loader',
      dataset: '$dataset',
      num_workers: 2,
      batch_size: 128,
      // persistent_workers: false,
    },
  },
}
