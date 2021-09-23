from typing import Optional
from flame.next_version.arguments import BaseArgs
from dataclasses import dataclass
from flame.next_version.distributed_training import start_distributed_training
from icecream import ic
import logging
import typed_args as ta

_logger = logging.getLogger(__name__)


@dataclass
class Args(BaseArgs):
    pretrained: Optional[str] = ta.add_argument(
        '-p', '--pretrained', help='pretrained checkpoint'
    )
    resume: Optional[str] = ta.add_argument(
        '-r', '--resume', help='恢复训练'
    )


def main_worker(args: Args):
    _logger.info('this is main worker')


def main():

    args = Args.from_args()
    # ic(args.parse_config())
    start_distributed_training(args)


if __name__ == '__main__':
    main()
