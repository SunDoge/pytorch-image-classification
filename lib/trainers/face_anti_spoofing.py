from typing import Tuple

from torch.functional import Tensor
from flame.state import BaseState
from torch import nn
from lib.arguments.training_args import Args
import logging
from torch.utils.data import DataLoader
from flame.helpers.tensorboard import Rank0SummaryWriter
from flame import helpers


_logger = logging.getLogger(__name__)


class State(BaseState):

    def __init__(
        self,
        model: nn.Module,
    ) -> None:
        super().__init__()
        self.model = model

    def get_batch_size(self, batch: Tuple[Tensor, Tensor]) -> int:
        return batch[0].size(0)


class Trainer:

    def __init__(
        self,
        args: Args,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        summary_writer: Rank0SummaryWriter,
        max_epochs: int,
    ) -> None:

        self.args = args
        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.summary_writer = summary_writer

        self.max_epochs = max_epochs

    def train_epoch(self, state: State, loader: DataLoader):
        pass

    def val_epoch(self, state: State, loader: DataLoader):
        pass

    def run(self, state: State):

        for epoch in state.epoch_wrapper(self.max_epochs):
            self.train_epoch(state, self.train_loader)
            self.val_epoch(state, self.val_loader)

            helpers.checkpoint_saver.save_checkpoint(
                state,
                self.args.experiment_dir,
            )


def main_worker(
    args: Args,
    model_config: dict,
    train_config: dict,
    test_config: dict,
):
    _logger.info(args)

    model = helpers.create_model_from_config(model_config)
    base_model = model.module

    
