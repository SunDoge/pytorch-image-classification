{
  cifar10(root, train, download=true):: {
    _name: 'lib.datasets.cifar.Cifar10',
    root: root,
    train: train,  // bool,
    transform: null,
    download: download,
  },
  cifar100: {

  },
}
