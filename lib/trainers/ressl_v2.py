

from typing import Iterable, List, Tuple


from torch import Tensor
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

from flame.next_version.state import BaseState
from flame.next_version import helpers
from train import Args
import logging
from flame.pytorch.meters.average_meter import DynamicAverageMeterGroup

from flame.pytorch.metrics.functional import topk_accuracy
from flame.next_version.helpers.tensorboard import Rank0SummaryWriter
from flame.next_version.helpers.checkpoint_saver import save_checkpoint
from .ressl import adjust_learning_rate

_logger = logging.getLogger(__name__)


class State(BaseState):

    def __init__(self,
                 model,
                 optimizer: torch.optim.SGD,
                 criterion,
                 ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def get_batch_size(self, batch: List[Tensor]) -> int:
        return batch[1].size(0)

    def adjust_learning_rate(self, base_lr: float, max_epochs: int):
        adjust_learning_rate(
            self.optimizer,
            self.epoch,
            base_lr,
            self.batch_idx,
            self.epoch_length,
            max_epochs,
            warm_up=5,
        )


def main_worker(
    args: Args,
    train_config: dict,
    max_epochs: int,
    model_config: dict,
    optimizer_config: dict,
    print_freq: int,
    criterion_config: dict,
):
    train_loader: DataLoader = helpers.create_data_loader_from_config(
        train_config
    )

    model: nn.Module = helpers.create_model_from_config(model_config)

    param_dict = {}
    for k, v in model.named_parameters():
        param_dict[k] = v

    bn_params = [v for n, v in param_dict.items() if (
        'bn' in n or 'bias' in n)]
    rest_params = [v for n, v in param_dict.items() if not (
        'bn' in n or 'bias' in n)]

    params = [{'params': bn_params, 'weight_decay': 0, },
              {'params': rest_params, 'weight_decay': 1e-4}]

    optimizer: torch.optim.SGD = helpers.create_optimizer_from_config(
        optimizer_config, params
    )
    base_lr = helpers.optimizer.get_learning_rate_from_optimizer(
        optimizer
    )

    criterion = helpers.create_from_config(criterion_config)

    # _logger.info(model)
    _logger.info(len(train_loader))

    state = State(
        model,
        optimizer,
        criterion
    )
    summary_writer = Rank0SummaryWriter(
        log_dir=args.experiment_dir
    )
    trainer = Trainer(
        args,
        train_loader,
        summary_writer,
        print_freq,
        max_epochs,
        base_lr
    )
    trainer.run(state)


class Trainer:

    def __init__(
        self,
        args: Args,
        train_loader: DataLoader,
        summary_writer: Rank0SummaryWriter,
        print_freq: int,
        max_epochs: int,
        base_lr: float,
    ) -> None:
        # 全局变量
        self.print_freq = print_freq
        self.max_epochs = max_epochs
        self.summary_writer = summary_writer
        self.args = args
        self.base_lr = base_lr
        self.train_loader = train_loader
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    def train(self, state: State, loader: DataLoader):
        meters = DynamicAverageMeterGroup()
        state.train()

        for batch, batch_size in state.iter_wrapper(loader):
            state.adjust_learning_rate(self.base_lr, self.max_epochs)

            loss = self.forward_model(state, batch, batch_size, meters)
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
        lr = helpers.optimizer.get_learning_rate_from_optimizer(
            state.optimizer)
        self.summary_writer.add_scalar(
            f'{prefix}/lr', lr, state.epoch
        )

    def forward_model(self, state: State, batch: Tuple[List[Tensor], Tensor], batch_size: int, meters: DynamicAverageMeterGroup):
        [img1, img2], _ = batch
        img1 = img1.to(self.device, non_blocking=True)
        img2 = img2.to(self.device, non_blocking=True)

        logitsq, logitsk = state.model(img1, img2)
        loss: Tensor = state.criterion(logitsq, logitsk)

        meters.update('loss', loss.item(), n=batch_size)

        return loss

    def run(self, state: State):
        for _ in state.epoch_wrapper(self.max_epochs):
            self.train(state, self.train_loader)

            save_checkpoint(
                state.state_dict(),
                self.args.experiment_dir
            )

            _logger.info(state.metrics)
            _logger.info(state.epoch_eta)

            if self.args.debug:
                break
