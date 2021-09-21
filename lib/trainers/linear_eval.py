from .supervised import Trainer as BaseTrainer, State as BaseState
from train import Args
from flame.next_version.helpers.checkpoint_saver import save_checkpoint
from flame.next_version import helpers
import torch
from torch import nn
import logging
from flame.next_version.helpers.tensorboard import Rank0SummaryWriter
from torch.utils.data import DataLoader

_logger = logging.getLogger(__name__)


class State(BaseState):

    def train(self, mode=True):
        super().train(mode=mode)
        # when finetuning last layer, always set eval for model
        self.model.eval()


class Trainer(BaseTrainer):

    def __init__(self, args: Args, train_config: dict, val_config: dict, max_epochs: int, model_config: dict, optimizer_config: dict, print_freq: int, criterion_config: dict, scheduler_config: dict) -> None:
        super().__init__(args, train_config, val_config, max_epochs, model_config, optimizer_config, print_freq, criterion_config, scheduler_config)
        