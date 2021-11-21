local ds = import '../datasets/celeba_spoof.libsonnet';

{
    local root = self,
    _name: 'lib.trainers.face_anti_spoofing.main_worker',
    args: '$args',

    model_config: {
        _name: 'timm.create_model',
        model_name: 'resnet18',
        pretrained: true,
        num_classes: 2, // binary classification
    },
    train_config: {

    },
    test_config: {

    },
}