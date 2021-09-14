from flame.next_version.arguments import BaseArgs
from dataclasses import dataclass
from flame.next_version.distributed_training import start_distributed_training
from icecream import ic
import logging

_logger = logging.getLogger(__name__)


@dataclass
class Args(BaseArgs):
    pass


def main_worker(args: Args):
    _logger.info('this is main worker')


def main():

    args = Args.from_args()
    ic(args)
    start_distributed_training(args)


if __name__ == '__main__':
    main()
