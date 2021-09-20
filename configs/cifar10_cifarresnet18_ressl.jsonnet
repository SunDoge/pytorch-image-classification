local model = import './models/ressl.libsonnet';

{
  _name: 'lib.trainers.ressl.Trainer',
  args: '$args',
  model_config: model.small_ressl,
  criterion_config: {
    _name: 'lib.losses.ressl.ResslLoss',
    T_student: 0.04,
    T_teacher: 0.1,
  },
}
