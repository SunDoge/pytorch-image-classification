{
  cifar10(root, train, download=true):: {
    _name: 'lib.datasets.cifar.Cifar10',
    root: root,
    train: train,  // bool,
    transform: '$transform',
    download: download,
    num_classes:: 10,
  },
  cifar100: {
    num_classes:: 100,
  },
}
