from flame.next_version.engine import BaseEngine, BaseModule, DataModule
import logging

_logger = logging.getLogger(__name__)


class MyModule(BaseModule):

    pass


class Engine(BaseEngine):

    # ProviderModule = BaseModule
    # class ProviderModule(BaseModule):
    #     pass

    def __init__(
        self
    ) -> None:
        super().__init__()

    def run(self, data_module: DataModule):
        
        for data in data_module.train_loader:
            _logger.info(data)
            break
