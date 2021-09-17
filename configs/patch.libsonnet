{
  setlr(lr):: {
    optimizer_config+: {
      lr: lr,
    },
  },
  mnist: {
    dataset: {
      _name: 'mnist',
    },
  },
}
