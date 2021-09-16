from flame.next_version.engine import BaseEngine, BaseModule, BaseState, DataModule
import logging
from flame.next_version.symbols import IConfig
from injector import inject

_logger = logging.getLogger(__name__)


class MyModule(BaseModule):

    pass


@inject
class Engine(BaseEngine):

    ProviderModule = MyModule
    # class ProviderModule(BaseModule):
    #     pass

    def __init__(
        self,
        config: IConfig
    ) -> None:
        super().__init__(config)
        _logger.info(config)
