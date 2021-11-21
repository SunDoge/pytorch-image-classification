from dataclasses import dataclass
import typed_args as ta
from flame.arguments import BaseArgs
from typing import Optional


@dataclass
class Args(BaseArgs):
    pretrained: Optional[str] = ta.add_argument(
        '-p', '--pretrained', help='pretrained checkpoint'
    )
    resume: Optional[str] = ta.add_argument(
        '-r', '--resume', help='恢复训练'
    )
