from typing import Tuple

from torch import Tensor
from flame.pytorch.meters.average_meter import DynamicAverageMeterGroup
from flame.next_version import helpers
from train import Args

import logging

from icecream import ic
from flame.next_version.state import BaseState
from torch import nn
import torch
from flame.pytorch.metrics.functional import topk_accuracy
from flame.next_version.helpers.tensorboard import Rank0SummaryWriter
from flame.next_version.helpers.checkpoint_saver import LatestCheckpointSaver, BestCheckpointSaver

_logger = logging.getLogger(__name__)


# class Engine:

#     def __init__(self, args: Args) -> None:
#         config = args.parse_config()
#         train_loader = create_data_loader_from_config(config['train'])
#         val_loader = create_data_loader_from_config(config['val'])
#         model = create_model_from_config(config['model'])
#         optimizer = helpers.create_optimizer_from_config(config['optimizer'], model.parameters())


class State(BaseState):

    def __init__(
        self,
        model,
        optimizer: torch.optim.SGD,
        device,
        criterion,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion
        self.best_acc1 = 0.

    def get_batch_size(self, batch) -> int:
        return batch[1].size(0)


def main_worker(
    args: Args,
    train_config: dict,
    val_config: dict,
    max_epochs: int,
    model_config: dict,
    optimizer_config: dict,
    print_freq: int,
):
    train_loader = helpers.create_data_loader_from_config(
        train_config
    )
    val_loader = helpers.create_data_loader_from_config(
        val_config
    )
    model = helpers.create_model_from_config(
        model_config
    )
    optimizer = helpers.create_optimizer_from_config(
        optimizer_config, model.parameters()
    )
    criterion = torch.nn.CrossEntropyLoss()
    _logger.info(len(train_loader))
    _logger.info(len(val_loader))

    state = State(
        model,
        optimizer,
        torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        criterion
    )
    summary_writer = Rank0SummaryWriter(
        log_dir=args.experiment_dir
    )

    for _ in state.epoch_wrapper(max_epochs):
        train(state, train_loader, print_freq)
        validate(state, val_loader, print_freq)
        _logger.info(state.epoch_eta)


def train(state: State, loader, print_freq: int):
    meters = DynamicAverageMeterGroup()
    state.train()
    loader.sampler.set_epoch(state.epoch)
    for batch, batch_size in state.iter_wrapper(loader):

        loss = forward_model(state, batch, batch_size, meters)
        state.optimizer.zero_grad()
        loss.backward()
        state.optimizer.step()

        if state.every_n_iters(n=print_freq):
            _logger.info(
                f'Train {state.iter_eta}\t{meters}'
            )


@torch.no_grad()
def validate(state: State, loader, print_freq: int):
    meters = DynamicAverageMeterGroup()
    state.eval()
    for batch, batch_size in state.iter_wrapper(loader):

        loss = forward_model(state, batch, batch_size, meters)

        if state.every_n_iters(n=print_freq):
            _logger.info(
                f'Val {state.iter_eta}\t{meters}'
            )

    meters.sync()
    return meters['acc1'].avg


def forward_model(state: State, batch: Tuple[Tensor, Tensor], batch_size: int, meters: DynamicAverageMeterGroup):
    image, target = batch
    image = image.to(state.device, non_blocking=True)
    target = target.to(state.device, non_blocking=True)

    output = state.model(image)
    loss: Tensor = state.criterion(output, target)

    acc1, acc5 = topk_accuracy(output, target, topk=(1, 5))

    meters.update('acc1', acc1.item(), n=batch_size)
    meters.update('acc5', acc5.item(), n=batch_size)
    meters.update('loss', loss.item(), n=batch_size)

    return loss
