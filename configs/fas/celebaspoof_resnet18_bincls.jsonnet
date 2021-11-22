local ds = import '../datasets/celeba_spoof.libsonnet';
local optim = import '../optimizers.libsonnet';

{
    local root = self,
    _name: 'lib.trainers.face_anti_spoofing.main_worker',
    args: '$args',
    max_epochs: 50,
    batch_size:: 128,

    model_config: {
        _name: 'timm.create_model',
        model_name: 'resnet18',
        pretrained: true,
        num_classes: 2, // binary classification
    },
    optimizer_config: optim.SGD(lr=0.1),
    criterion_config: {
        _name: 'torch.nn.CrossEntropyLoss'
    },

    local ds_prefix = 'lib.datasets.celeba_spoof.',
    local ds_root_prefix = '/mnt/cephfs/dataset/FAS/CelebA_Spoof/CelebA_Spoof/Data/',
    local loader_name = 'flame.pytorch.helpers.create_data_loader',
    train_config: {
        transform: {
            _name: ds_prefix + 'TrainTransform',
        },
        dataset: ds.celeba_spoof(
            ds_root_prefix + 'train',
            split='train',
        ),
        loader: {
            _name: loader_name,
            dataset: '$dataset',
            num_workers: 4,
            batch_size: root.batch_size,
        },
    },
    test_config: {
        transform: {
            _name: ds_prefix + 'ValTransform',
        },
        dataset: ds.celeba_spoof(
            ds_root_prefix + 'test',
            split='test',
        ),
        loader: {
            local factor = 2,
            _name: loader_name,
            dataset: '$dataset',
            num_workers: 4 * factor,
            batch_size: root.batch_size * factor,
        },
    },
}