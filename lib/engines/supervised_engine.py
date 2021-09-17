from flame.next_version import helpers
from train import Args

import logging

from icecream import ic
from flame.next_version.state import BaseState
from torch import nn


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
        optimizer,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer

    def get_batch_size(self, batch) -> int:
        return batch[1].size(0)


def main_worker(
    args: Args,
    train_config: dict,
    val_config: dict,
    max_epochs: int,
    model_config: dict,
    optimizer_config: dict,

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
    _logger.info(len(train_loader))
    _logger.info(len(val_loader))

    state = State(
        model,
        optimizer
    )

    for _ in state.epoch_wrapper(max_epochs):
        train(state, train_loader)
        validate(state, val_loader)
        _logger.info(state.epoch_eta)


def train(state: State, loader):
    state.train()
    for batch, batch_size in state.iter_wrapper(loader):
        _logger.info(state.iter_eta)


def validate(state: State, loader):
    state.eval()
    for batch, batch_size in state.iter_wrapper(loader):
        _logger.info(state.iter_eta)
