from flame.next_version.engine import BaseEngine, BaseModule


class Engine(BaseEngine):

    # ProviderModule = BaseModule
    # class ProviderModule(BaseModule):
    #     pass

    def __init__(self) -> None:
        super().__init__()
