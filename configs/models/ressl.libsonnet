{
  local root = self,
  cifar_resnet(arch='resnet18'):: {
    _name: 'lib.models.ressl.cifar_resnet.cifar_resnet',
    arch: arch,
  },
  small_ressl: {
    _name: 'lib.models.ressl.ressl.ReSSL',
    backbone: root.cifar_resnet(),
    hidden_dim: 2048,
    dim: 128,
    dim_in: 512,
    K: 4096,
    m: 0.99,
  },
  medium_ressl: self.small_ressl {
    K: 16384,
    m: 0.996,
  },
}
