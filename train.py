from flame.next_version.config_parser import ConfigParser
from flame.next_version.config import from_file
from icecream import ic

cp = ConfigParser()

config = from_file('configs/000.jsonnet')

composed = cp.parse(config['train_transform'])

ic(composed)
