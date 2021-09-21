{
  local root = self,
  cifar_resnet(arch='resnet18'):: {
    _name: 'lib.models.ressl.cifar_resnet.cifar_resnet',
    arch: arch,
  },
  small_ressl: {
    _name: 'lib.models.ressl.ressl.ReSSL',
    backbone: root.cifar_resnet(),
    hidden_dim: 512,
    dim: 128,
    dim_in: 10,
    K: 4096,
    m: 0.99,
  },
  medium_ressl: self.small_ressl {
    K: 16384,
    m: 0.996,
  },
}
