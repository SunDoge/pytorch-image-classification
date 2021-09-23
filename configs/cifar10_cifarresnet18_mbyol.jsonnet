// Ressl on CIFAR10
local ds = import './datasets/cifar.libsonnet';
local models = import './models/ressl.libsonnet';
local optimizers = import './optimizers.libsonnet';

{
  local root = self,

  _name: 'lib.trainers.ressl.Trainer',
  args: '$args',

  max_epochs: 200,
  print_freq: 50,
  learning_rate:: 0.06,
  batch_size:: 256,

  model_config: {
    _name: 'lib.models.byol.mbyol.Mbyol',
    online_network: models.cifar_resnet(),
    target_network: models.cifar_resnet(),
  },

  optimizer_config: optimizers.SGD(
    // normalize lr
    root.learning_rate * root.batch_size / 256
  ),
  criterion_config: {
    _name: 'torch.nn.BCEWithLogitsLoss',
  },

  train_config: {
    transform: {
      _name: 'lib.datasets.presets.ResslCifarTransform',
    },
    dataset: ds.cifar10pair(
      './data/cifar10',
      train=true,
    ),
    loader: {
      _name: 'flame.pytorch.helpers.create_data_loader',
      dataset: '$dataset',
      num_workers: 4,
      batch_size: root.batch_size,
      drop_last: true,
    },
  },

}
