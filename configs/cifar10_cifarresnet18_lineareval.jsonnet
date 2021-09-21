// Train ResNet20 on CIFAR10
local ds = import './datasets/cifar.libsonnet';
local models = import './models/ressl.libsonnet';
local optimizers = import './optimizers.libsonnet';
local schedulers = import './schedulers.libsonnet';
local trans = import './transforms/cifar.libsonnet';

{
  local root = self,
  // main_worker: 'lib.engines.supervised_engine.Engine',
  _name: 'lib.trainers.linear_eval.main_worker',
  args: '$args',
  max_epochs: 100,
  print_freq: 10,
  learning_rate:: 30.0,
  batch_size:: 256,

  model_config: models.cifar_resnet(),
  optimizer_config: optimizers.SGD(
    // normalize lr
    root.learning_rate * root.batch_size / 256,
    weight_decay=0
  ),
  scheduler_config: schedulers.MultiStepLR([60, 80]),
  criterion_config: {
    _name: 'torch.nn.CrossEntropyLoss',
  },

  train_config: {
    transform: trans.supervised.train,
    dataset: ds.cifar10(
      './data/cifar10',
      train=true,
    ),
    loader: {
      _name: 'flame.pytorch.helpers.create_data_loader',
      dataset: '$dataset',
      num_workers: 4,
      batch_size: root.batch_size,
      // persistent_workers: false,
    },
  },
  val_config: {
    transform: trans.supervised.val,
    dataset: ds.cifar10(
      './data/cifar10',
      train=false,
    ),
    loader: {
      _name: 'flame.pytorch.helpers.create_data_loader',
      dataset: '$dataset',
      num_workers: 4,
      batch_size: root.batch_size * 2,
      // persistent_workers: false,
    },
  },
}
