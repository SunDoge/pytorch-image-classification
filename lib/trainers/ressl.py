from flame.next_version.state import BaseState
from flame.next_version import helpers
from train import Args
import logging

_logger = logging.getLogger(__name__)


class State(BaseState):

    def __init__(self) -> None:
        super().__init__()


class Trainer:

    def __init__(
        self,
        args: Args,
        model_config: dict,
        criterion_config: dict
    ) -> None:

        model = helpers.create_model_from_config(model_config)
        criterion = helpers.create_from_config(criterion_config)

        _logger.info(model)
