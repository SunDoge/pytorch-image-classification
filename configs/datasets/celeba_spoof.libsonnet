{
    celeba_spoof(root, split = 'train'):: {
        _name: 'lib.datasets.celeba_spoof.CelebASpoof',
        root: root,
        file_list: 'data/celeba_spoof/%s_list.pkl' % split,
        transform: '$transform',
    },
}