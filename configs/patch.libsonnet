{
  setlr(lr):: {
    optimizer+: {
      lr: lr,
    },
  },
  mnist: {
    dataset: {
      _name: 'mnist',
    },
  },
}
