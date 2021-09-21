from os import stat
from typing import Iterable, Tuple
import math

from torch import Tensor
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from flame.next_version.state import BaseState
from flame.next_version import helpers
from train import Args
import logging
from flame.pytorch.meters.average_meter import DynamicAverageMeterGroup

from flame.pytorch.metrics.functional import topk_accuracy
from flame.next_version.helpers.tensorboard import Rank0SummaryWriter
from flame.next_version.helpers.checkpoint_saver import save_checkpoint

_logger = logging.getLogger(__name__)


class State(BaseState):

    def __init__(self,
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

    def get_batch_size(self, batch) -> int:
        return batch[1].size(0)

class Trainer:

    def __init__(
        self,
        args: Args,
        train_config: dict,
        max_epochs: int,
        model_config: dict,
        optimizer_config: dict,
        print_freq: int,
        criterion_config: dict,
    ) -> None:

        train_loader: DataLoader = helpers.create_data_loader_from_config(
            train_config
        )

        model: nn.Module = helpers.create_model_from_config(model_config)
        optimizer: torch.optim.SGD = helpers.create_optimizer_from_config(
            optimizer_config, model.parameters()
        )

        criterion = helpers.create_from_config(criterion_config)

        # _logger.info(model)
        _logger.info(len(train_loader))

        state = State(
            model,
            optimizer,
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

            save_checkpoint(state.state_dict(),
                            args.experiment_dir)

            _logger.info(state.metrics)
            _logger.info(state.epoch_eta)

            if args.debug:
                break

    def train(self, state: State, loader: DataLoader):
        meters = DynamicAverageMeterGroup()
        state.train()

        for batch, batch_size in state.iter_wrapper(loader):
            adjust_learning_rate(state, self.max_epochs)
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
    
    pass


def forward_model(state: State, batch: Tuple[Tensor, Tensor], batch_size: int, meters: DynamicAverageMeterGroup):
    [img1, img2], _ = batch
    img1 = img1.to(state.device, non_blocking=True)
    img2 = img2.to(state.device, non_blocking=True)

    logitsq, logitsk = state.model(img1, img2)
    loss: Tensor = state.criterion(logitsq, logitsk)

    meters.update('loss', loss.item(), n=batch_size)

    return loss

def adjust_learning_rate(state:State,  epochs, warm_up=5):
    base_lr = helpers.optimizer.get_learning_rate_from_optimizer(state.optimizer)
    iteration_per_epoch = state.epoch_length

    T = state.epoch * iteration_per_epoch + state.batch_idx
    warmup_iters = warm_up * iteration_per_epoch
    total_iters = (epochs - warm_up) * iteration_per_epoch

    if state.epoch < warm_up:
        lr = base_lr * 1.0 * T / warmup_iters
    else:
        T = T - warmup_iters
        lr = 0.5 * base_lr * (1 + math.cos(1.0 * T / total_iters * math.pi))
    
    for param_group in state.optimizer.param_groups:
        param_group['lr'] = lr