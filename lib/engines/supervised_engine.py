from flame.next_version import helpers
from train import Args
from flame.next_version.engine import BaseEngine, BaseModule, BaseState, DataModule
import logging
from flame.next_version.symbols import IConfig
from injector import inject
from icecream import ic
from flame.next_version.helpers.data import create_data_loader_from_config
from flame.next_version.helpers.model import create_model_from_config

_logger = logging.getLogger(__name__)


class Engine:

    def __init__(self, args: Args) -> None:
        config = args.parse_config()
        train_loader = create_data_loader_from_config(config['train'])
        val_loader = create_data_loader_from_config(config['val'])
        model = create_model_from_config(config['model'])
        optimizer = helpers.create_optimizer_from_config(config['optimizer'], model.parameters())


        
        


    
