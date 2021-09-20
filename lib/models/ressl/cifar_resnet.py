from torchvision.models import resnet
from torch import nn


def cifar_resnet(arch: str, num_classes: int = 10):
    resnet_arch = getattr(resnet, arch)
    net: resnet.ResNet = resnet_arch(num_classes=num_classes)
    net.conv1 = nn.Conv2d(
        3, 64,
        kernel_size=3, stride=1,
        padding=1, bias=False
    )
    net.maxpool = nn.Identity()
    return net
