from typing import Tuple

from torch.functional import Tensor
from flame.state import BaseState
from torch import nn
from lib.arguments.training_args import Args
import logging

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


def main_worker(
    args: Args,
):
    _logger.info(args)
