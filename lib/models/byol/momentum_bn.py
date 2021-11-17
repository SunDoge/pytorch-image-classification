from torch import nn
import torch
import torch.nn.functional as F
import math
import logging

_logger = logging.getLogger(__name__)


class SimpleBatchNorm3d(nn.BatchNorm3d):
    def forward(self, x):
        if self.train:
            self.running_mean = self.running_mean.clone()
            self.running_var = self.running_var.clone()
            result = super(SimpleBatchNorm3d, self).forward(x)
        return result


class SimpleBatchNorm2d(nn.BatchNorm2d):
    def forward(self, x):
        if self.train:
            self.running_mean = self.running_mean.clone()
            self.running_var = self.running_var.clone()
            result = super(SimpleBatchNorm2d, self).forward(x)
        return result


class SimpleBatchNorm1d(nn.BatchNorm1d):
    def forward(self, x):
        if self.train:
            self.running_mean = self.running_mean.clone()
            self.running_var = self.running_var.clone()
            result = super(SimpleBatchNorm1d, self).forward(x)
        return result


class MomentumBatchNorm3d(nn.BatchNorm3d):
    def __init__(self, num_features, eps=1e-5, momentum=1.0, affine=True, track_running_stats=True, total_iters=100):
        super(MomentumBatchNorm3d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.total_iters = total_iters
        self.cur_iter = 0
        self.mean_last_batch = None
        self.var_last_batch = None

    def momentum_cosine_decay(self):
        self.cur_iter += 1
        self.momentum = (
            math.cos(math.pi * (self.cur_iter / self.total_iters)) + 1) * 0.5

    def forward(self, x):
        # if not self.training:
        #     return super().forward(x)

        # Changed
        mean = torch.mean(x, dim=[0, 2, 3, 4])
        var = torch.var(x, dim=[0, 2, 3, 4])
        n = x.numel() / x.size(1)

        with torch.no_grad():
            tmp_running_mean = self.momentum * mean + \
                (1 - self.momentum) * self.running_mean
            # update running_var with unbiased var
            tmp_running_var = self.momentum * var * n / \
                (n - 1) + (1 - self.momentum) * self.running_var

        # Changed
        x = (x - tmp_running_mean[None, :, None, None, None].detach()) / (
            torch.sqrt(tmp_running_var[None, :,
                                       None, None, None].detach() + self.eps)
        )
        if self.affine:
            x = x * self.weight[None, :, None, None, None] + \
                self.bias[None, :, None, None, None]

        # update the parameters
        if self.mean_last_batch is None and self.var_last_batch is None:
            self.mean_last_batch = mean
            self.var_last_batch = var
        else:
            self.running_mean = (
                self.momentum * ((mean + self.mean_last_batch) * 0.5) +
                (1 - self.momentum) * self.running_mean
            )
            self.running_var = (
                self.momentum * ((var + self.var_last_batch)
                                 * 0.5) * n / (n - 1)
                + (1 - self.momentum) * self.running_var
            )
            self.mean_last_batch = None
            self.var_last_batch = None
            self.momentum_cosine_decay()

        return x


class MomentumBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=1.0, affine=True, track_running_stats=True, total_iters=100):
        super(MomentumBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.total_iters = total_iters
        self.cur_iter = 0
        self.mean_last_batch = None
        self.var_last_batch = None

    def momentum_cosine_decay(self):
        self.cur_iter += 1
        self.momentum = (
            math.cos(math.pi * (self.cur_iter / self.total_iters)) + 1) * 0.5

    def forward(self, x):
        # if not self.training:
        #     return super().forward(x)

        mean = torch.mean(x, dim=[0, 2, 3])
        var = torch.var(x, dim=[0, 2, 3])
        n = x.numel() / x.size(1)

        with torch.no_grad():
            tmp_running_mean = self.momentum * mean + \
                (1 - self.momentum) * self.running_mean
            # update running_var with unbiased var
            tmp_running_var = self.momentum * var * n / \
                (n - 1) + (1 - self.momentum) * self.running_var

        x = (x - tmp_running_mean[None, :, None, None].detach()) / (
            torch.sqrt(tmp_running_var[None, :,
                                       None, None].detach() + self.eps)
        )
        if self.affine:
            x = x * self.weight[None, :, None, None] + \
                self.bias[None, :, None, None]

        # update the parameters
        if self.mean_last_batch is None and self.var_last_batch is None:
            self.mean_last_batch = mean
            self.var_last_batch = var
        else:
            self.running_mean = (
                self.momentum * ((mean + self.mean_last_batch) * 0.5) +
                (1 - self.momentum) * self.running_mean
            )
            self.running_var = (
                self.momentum * ((var + self.var_last_batch)
                                 * 0.5) * n / (n - 1)
                + (1 - self.momentum) * self.running_var
            )
            self.mean_last_batch = None
            self.var_last_batch = None
            self.momentum_cosine_decay()

        return x


class MomentumBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=1.0, affine=True, track_running_stats=True, total_iters=100):
        super(MomentumBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.total_iters = total_iters
        self.cur_iter = 0
        self.mean_last_batch = None
        self.var_last_batch = None

    def momentum_cosine_decay(self):
        self.cur_iter += 1
        self.momentum = (
            math.cos(math.pi * (self.cur_iter / self.total_iters)) + 1) * 0.5

    def forward(self, x):
        # if not self.training:
        #     return super().forward(x)

        mean = torch.mean(x, dim=[0])
        var = torch.var(x, dim=[0])
        n = x.numel() / x.size(1)

        with torch.no_grad():
            tmp_running_mean = self.momentum * mean + \
                (1 - self.momentum) * self.running_mean
            # update running_var with unbiased var
            tmp_running_var = self.momentum * var * n / \
                (n - 1) + (1 - self.momentum) * self.running_var

        x = (x - tmp_running_mean[None, :].detach()) / \
            (torch.sqrt(tmp_running_var[None, :].detach() + self.eps))
        if self.affine:
            x = x * self.weight[None, :] + self.bias[None, :]

        # update the parameters
        if self.mean_last_batch is None and self.var_last_batch is None:
            self.mean_last_batch = mean
            self.var_last_batch = var
        else:
            self.running_mean = (
                self.momentum * ((mean + self.mean_last_batch) * 0.5) +
                (1 - self.momentum) * self.running_mean
            )
            self.running_var = (
                self.momentum * ((var + self.var_last_batch)
                                 * 0.5) * n / (n - 1)
                + (1 - self.momentum) * self.running_var
            )
            self.mean_last_batch = None
            self.var_last_batch = None
            self.momentum_cosine_decay()

        return x


def get_simple_bn(module: nn.Module):
    if isinstance(module, nn.BatchNorm3d):
        output_module = SimpleBatchNorm3d(
            module.num_features,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            track_running_stats=module.track_running_stats
        )
    elif isinstance(module, nn.BatchNorm2d):
        output_module = SimpleBatchNorm2d(
            module.num_features,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            track_running_stats=module.track_running_stats
        )
    elif isinstance(module, nn.BatchNorm1d):
        output_module = SimpleBatchNorm1d(
            module.num_features,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            track_running_stats=module.track_running_stats
        )

    _logger.debug('replace %s with %s', module, output_module)
    return output_module


def get_momentum_bn(module: nn.Module, total_iters: int):
    if isinstance(module, nn.BatchNorm3d):
        output_module = MomentumBatchNorm3d(
            module.num_features,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            track_running_stats=module.track_running_stats,
            total_iters=total_iters
        )
    elif isinstance(module, nn.BatchNorm2d):
        output_module = MomentumBatchNorm2d(
            module.num_features,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            track_running_stats=module.track_running_stats,
            total_iters=total_iters
        )
    elif isinstance(module, nn.BatchNorm1d):
        output_module = MomentumBatchNorm1d(
            module.num_features,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            track_running_stats=module.track_running_stats,
            total_iters=total_iters
        )

    _logger.debug('replace %s with %s', module, output_module)
    return output_module


def convert_simple_bn(module: nn.Module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = get_simple_bn(module)

        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias

        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    for name, child in module.named_children():
        module_output.add_module(name, convert_simple_bn(child))
    del module
    return module_output


def convert_momentum_bn(module: nn.Module, total_iters: int):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = get_momentum_bn(module, total_iters)

        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias

        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    for name, child in module.named_children():
        module_output.add_module(name, convert_momentum_bn(child, total_iters))
    del module
    return module_output


if __name__ == "__main__":
    simple_bn3d = SimpleBatchNorm3d(4)
    momentum_bn3d = MomentumBatchNorm3d(4)

    x = torch.rand(2, 4, 8, 8, 8)

    y1 = simple_bn3d(x)
    print('y1', y1.shape)
    y2 = momentum_bn3d(x)
    print('y2', y2.shape)
