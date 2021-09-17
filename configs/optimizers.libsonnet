{
  SGD(lr):: {
    _name: 'torch.optim.SGD',
    params: '$params',
    lr: lr,
    momentum: 0.9,
    weight_decay: 1e-4,
    nesterov: true,
  },
}
