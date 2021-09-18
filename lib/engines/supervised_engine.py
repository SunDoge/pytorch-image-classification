from typing import Iterable, Tuple

from torch import Tensor
from torch.utils.data.dataloader import DataLoader
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
from flame.next_version.helpers.checkpoint_saver import save_checkpoint

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
        scheduler: torch.optim.lr_scheduler.MultiStepLR,
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


class MainWorker:

    def __init__(
        self,
        args: Args,
        train_config: dict,
        val_config: dict,
        max_epochs: int,
        model_config: dict,
        optimizer_config: dict,
        print_freq: int,
        criterion_config: dict,
        scheduler_config: dict,
    ) -> None:
        train_loader = helpers.create_data_loader_from_config(
            train_config
        )
        val_loader = helpers.create_data_loader_from_config(
            val_config
        )
        model: nn.Module = helpers.create_model_from_config(
            model_config
        )
        optimizer: torch.optim.SGD = helpers.create_optimizer_from_config(
            optimizer_config, model.parameters()
        )
        scheduler: torch.optim.lr_scheduler.MultiStepLR = helpers.create_scheduler_from_config(
            scheduler_config, optimizer
        )
        criterion: torch.nn.CrossEntropyLoss = helpers.create_from_config(
            criterion_config
        )
        _logger.info(len(train_loader))
        _logger.info(len(val_loader))

        state = State(
            model,
            optimizer,
            scheduler,
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            criterion
        )
        summary_writer = Rank0SummaryWriter(
            log_dir=args.experiment_dir
        )

        # 全局变量
        self.print_freq = print_freq
        self.max_epochs = max_epochs
        self.summary_writer = summary_writer
        self.args = args

        for _ in state.epoch_wrapper(max_epochs):
            self.train(state, train_loader)
            self.validate(state, val_loader)

            best_acc1 = state.metrics.get('best_acc1', 0.)
            acc1 = state.metrics['acc1']
            is_best = acc1 > best_acc1
            if is_best:
                state.metrics['best_acc1'] = acc1

            save_checkpoint(state.state_dict(),
                            args.experiment_dir, is_best=is_best)

            _logger.info(state.epoch_eta)

            if args.debug:
                break

    def train(self, state: State, loader: DataLoader):
        meters = DynamicAverageMeterGroup()
        state.train()

        for batch, batch_size in state.iter_wrapper(loader):

            loss = forward_model(state, batch, batch_size, meters)
            state.optimizer.zero_grad()
            loss.backward()
            state.optimizer.step()

            if state.every_n_iters(n=self.print_freq):
                _logger.info(
                    f'Train {state.iter_eta}\t{meters}'
                )
                if self.args.debug:
                    break

        meters.sync()

        _logger.info(
            f'Train complete [{state.epoch}/{self.max_epochs}]\t{meters}'
        )
        self.write_summary(
            state, meters, 'train'
        )

    def write_summary(self, state: State, meters: DynamicAverageMeterGroup, prefix: str):
        self.summary_writer.add_scalar(
            f'{prefix}/loss', meters['loss'].avg, state.epoch
        )
        self.summary_writer.add_scalar(
            f'{prefix}/acc1', meters['acc1'].avg, state.epoch
        )
        self.summary_writer.add_scalar(
            f'{prefix}/acc5', meters['acc5'].avg, state.epoch
        )

    @torch.no_grad()
    def validate(self, state: State, loader: DataLoader):
        meters = DynamicAverageMeterGroup()
        state.eval()
        for batch, batch_size in state.iter_wrapper(loader):

            _loss = forward_model(state, batch, batch_size, meters)

            if state.every_n_iters(n=self.print_freq):
                _logger.info(
                    f'Val {state.iter_eta}\t{meters}'
                )
                if self.args.debug:
                    break

        meters.sync()

        _logger.info(
            f'Val complete [{state.epoch}/{self.max_epochs}]\t{meters}'
        )

        self.write_summary(state, meters, 'val')

        state.metrics['acc1'] = meters['acc1'].avg


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

        if args.debug:
            break


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
