{
  resnet(num_classes, depth):: {
    _name: 'torchvision.models.resnet%s' % depth,
    num_classes: num_classes,
  },
  resnet18(num_classes):: self.resnet(num_classes, 18),
  resnet50(num_classes):: self.resnet(num_classes, 50),
}
