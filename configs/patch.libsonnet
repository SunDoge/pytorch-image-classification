{
  setlr(lr):: {
    optimizer_config+: {
      lr: lr,
    },
  },
  setmaxepoch(epoch):: {
    max_epochs: epoch,
  },
  mnist: {
    dataset: {
      _name: 'mnist',
    },
  },
}
