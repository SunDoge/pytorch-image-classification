from typing import Tuple

from torch.functional import Tensor
from flame.pytorch.meters.average_meter import DynamicAverageMeterGroup
from flame.state import BaseState
from torch import nn
from lib.arguments.training_args import Args
import logging
from torch.utils.data import DataLoader
from flame.helpers.tensorboard import Rank0SummaryWriter
from flame import helpers
import torch

_logger = logging.getLogger(__name__)


class State(BaseState):

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def get_batch_size(self, batch: Tuple[Tensor, Tensor]) -> int:
        return batch[0].size(0)


class Trainer:

    def __init__(
        self,
        args: Args,
        train_loader: DataLoader,
        test_loader: DataLoader,
        summary_writer: Rank0SummaryWriter,
        max_epochs: int,
    ) -> None:

        self.args = args

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.summary_writer = summary_writer

        self.max_epochs = max_epochs
        self.device = args.device

    def train_epoch(self, state: State, loader: DataLoader):
        state.train()
        meters = DynamicAverageMeterGroup()
        for batch, batch_size in state.iter_wrapper(loader):
            loss = self.forward_step(state, batch, batch_size, meters)
            state.optimizer.zero_grad()
            loss.backward()
            state.optimizer.step()

            if state.every_n_iters(10):
                _logger.info(
                    f'train {state.iter_eta} {meters}'
                )
                if self.args.debug:
                    break

        meters.sync()
        _logger.info(
            f'train complete, {state.epoch}/{self.max_epochs} {meters}')

        meters.write_summary(self.summary_writer, prefix='train')

    @torch.no_grad()
    def val_epoch(self, state: State, loader: DataLoader):
        state.eval()

        meters = DynamicAverageMeterGroup()
        for batch, batch_size in state.iter_wrapper(loader):
            _loss = self.forward_step(state, batch, batch_size, meters)

            if state.every_n_iters(10):
                _logger.info(
                    f'val {state.iter_eta} {meters}'
                )
                if self.args.debug:
                    break

        meters.sync()
        _logger.info(
            f'val complete, {state.epoch}/{self.max_epochs} {meters}')

        meters.write_summary(self.summary_writer, prefix='val')

    def forward_step(
        self,
        state: State,
        batch: Tuple[Tensor, Tensor],
        batch_size: int,
        meters: DynamicAverageMeterGroup
    ):
        image, label = batch
        image = image.to(self.device, non_blocking=True)
        label = label.to(self.device, non_blocking=True)

        output = state.model(image)
        loss: Tensor = state.criterion(output, label)

        meters.update('loss', loss.item(), n=batch_size)

        return loss

    def run(self, state: State):

        for epoch in state.epoch_wrapper(self.max_epochs):
            self.train_epoch(state, self.train_loader)
            self.val_epoch(state, self.test_loader)

            helpers.checkpoint_saver.save_checkpoint(
                state.state_dict(),
                self.args.experiment_dir,
            )

            if self.args.debug:
                break


def main_worker(
    args: Args,
    model_config: dict,
    train_config: dict,
    test_config: dict,
    optimizer_config: dict,
    criterion_config: dict,
    max_epochs: int,
):
    _logger.info(args)

    base_model = helpers.create_model_from_config(model_config)
    model = helpers.prepare_model(base_model, args.device)
    optimizer = helpers.create_optimizer_from_config(
        optimizer_config, model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=1e-4,
    )

    summary_writer = Rank0SummaryWriter(log_dir=args.experiment_dir)

    train_loader = helpers.create_data_loader_from_config(train_config)
    test_loader = helpers.create_data_loader_from_config(test_config)
    criterion = helpers.create_from_config(criterion_config)

    state = State(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    trainer = Trainer(
        args=args,
        train_loader=train_loader,
        test_loader=test_loader,
        summary_writer=summary_writer,
        max_epochs=max_epochs,
    )

    trainer.run(state)
