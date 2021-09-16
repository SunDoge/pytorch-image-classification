{
  cifar_resnet(num_classes, depth):: {
    _name: 'lib.models.cifar_resnet.cifar_resnet%s' % depth,
    num_classes: num_classes,
  },
  cifar_resnet20(num_classes):: self.cifar_resnet(num_classes, 20),
}
