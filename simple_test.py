
from flame.next_version.config import parse_config, from_snippet
from icecream import ic
import rich
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('fuck')
args = parser.parse_args()
print(args)

snippet = parse_config(
    'configs/001.jsonnet',
    ['set_lr(0.1)', 'mnist']
)

json_str = from_snippet(snippet)

rich.print(json_str)
