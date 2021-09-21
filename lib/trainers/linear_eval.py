from .supervised import Trainer, State as BaseState
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


def main_worker(
    args: Args,
    train_config: dict,
    val_config: dict,
    max_epochs: int,
    model_config: dict,
    optimizer_config: dict,
    print_freq: int,
    criterion_config: dict,
    scheduler_config: dict,
):
    train_loader: DataLoader = helpers.create_data_loader_from_config(
        train_config
    )
    val_loader = helpers.create_data_loader_from_config(
        val_config
    )
    model: nn.Module = helpers.create_model_from_config(
        model_config
    )
    # TODO load weights
    if args.weights:
        _logger.info('load weights from: %s', args.weights)
        cp = torch.load(args.weights, map_location='cpu')

    optimizer: torch.optim.SGD = helpers.create_optimizer_from_config(
        optimizer_config, model.parameters()
    )
    scheduler: torch.optim.lr_scheduler.MultiStepLR = helpers.create_scheduler_from_config(
        scheduler_config, optimizer
    )
    criterion: torch.nn.CrossEntropyLoss = helpers.create_from_config(
        criterion_config
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = State(
        model,
        optimizer,
        scheduler,
        device,
        criterion
    )
    summary_writer = Rank0SummaryWriter(
        log_dir=args.experiment_dir
    )
    trainer = Trainer(
        args,
        summary_writer,
        train_loader,
        val_loader,
        print_freq,
        max_epochs
    )
    trainer.run(state)
