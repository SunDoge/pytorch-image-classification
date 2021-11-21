{
    celeba_spoof(root, split = 'train'):: {
        root: root,
        file_list: 'data/celeba_spoof/%s_list.pkl' % split,
        transform: '$transform',
    },
}