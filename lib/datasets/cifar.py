from torch._C import _ImperativeEngine
from torchvision.datasets import CIFAR10, CIFAR100


def supervised_getitem(self: CIFAR10, index: int):
    img, target = self.data[index], self.targets[index]

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    # img = Image.fromarray(img)

    if self.transform is not None:
        img = self.transform(image=img)['image']

    if self.target_transform is not None:
        target = self.target_transform(target)

    return img, target


def unsupervised_getitem(self: CIFAR10, index: int):
    img, target = self.data[index], self.targets[index]
    # 要求transform输入一张图，输出多个view
    views: list = self.transform(img)
    return views, target


class Cifar10(CIFAR10):

    def __getitem__(self, index: int):
        return supervised_getitem(self, index)


class Cifar100(CIFAR100):

    def __getitem__(self, index: int):
        return supervised_getitem(self, index)


class Cifar10Pair(CIFAR10):

    def __getitem__(self, index: int):
        return unsupervised_getitem(self, index)
