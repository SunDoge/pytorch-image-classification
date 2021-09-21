{
  MultiStepLR(milestones=[100, 150]):: {
    _name: 'torch.optim.lr_scheduler.MultiStepLR',
    optimizer: '$optimizer',
    milestones: milestones,
  },
}
