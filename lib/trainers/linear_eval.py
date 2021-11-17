from torchvision.models.resnet import ResNet
from flame.next_version.helpers import model
from typing import Dict

from torch.functional import Tensor
from lib.models.ressl.head import LinearHead
from .supervised import Trainer, State as BaseState
from train import Args
from flame.next_version.helpers.checkpoint_saver import save_checkpoint
from flame.next_version import helpers
import torch
from torch import nn
import logging
from flame.next_version.helpers.tensorboard import Rank0SummaryWriter
from torch.utils.data import DataLoader
from pygtrie import CharTrie

_logger = logging.getLogger(__name__)


class State(BaseState):

    def train(self, mode=True):
        super().train(mode=mode)
        # when finetuning last layer, always set eval for model
        self.model.eval()


def main_worker(
    args: Args,
    train_config: dict,
    val_config: dict,
    max_epochs: int,
    model_config: dict,
    optimizer_config: dict,
    print_freq: int,
    criterion_config: dict,
    scheduler_config: dict,
):
    train_loader: DataLoader = helpers.create_data_loader_from_config(
        train_config
    )
    val_loader = helpers.create_data_loader_from_config(
        val_config
    )
    model = helpers.create_model_from_config(
        model_config,
        find_unused_parameters=True,
    )

    # freeze all layers but the last fc
    base_model: ResNet = model.module
    for name, param in base_model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    base_model.fc.weight.data.normal_(mean=0.0, std=0.01)
    base_model.fc.bias.data.zero_()

    # TODO load weights
    if args.pretrained:
        _logger.info('load weights from: %s', args.pretrained)
        cp = torch.load(args.pretrained, map_location='cpu')
        load_weights(base_model, cp['model'])

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2, len(parameters)  # fc.weight, fc.bias

    optimizer: torch.optim.SGD = helpers.create_optimizer_from_config(
        optimizer_config, parameters
    )
    scheduler: torch.optim.lr_scheduler.MultiStepLR = helpers.create_scheduler_from_config(
        scheduler_config, optimizer
    )
    criterion: torch.nn.CrossEntropyLoss = helpers.create_from_config(
        criterion_config
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = State(
        model,
        optimizer,
        scheduler,
        device,
        criterion
    )
    summary_writer = Rank0SummaryWriter(
        log_dir=args.experiment_dir
    )
    trainer = Trainer(
        args,
        summary_writer,
        train_loader,
        val_loader,
        print_freq,
        max_epochs
    )

    initial_state_dict = state_dict_to_cpu(model.state_dict())
    trainer.run(state)
    state_dict = state_dict_to_cpu(model.state_dict())
    sanity_check(state_dict, initial_state_dict)


def load_weights(base_model: nn.Module, state_dict: dict):
    state_trie = CharTrie(state_dict)

    ressl_prefix = 'module.encoder_q.net.'
    mbyol_prefix = 'module.online_network.'
    prefix = ''
    new_state_dict = {}
    if state_trie.has_subtrie(ressl_prefix):
        prefix = ressl_prefix
    elif state_trie.has_subtrie(mbyol_prefix):
        prefix = mbyol_prefix
    else:
        raise Exception()

    _logger.info('prefix=%s', prefix)
    
    for key, value in state_trie.items(prefix=prefix):
        new_key = key[len(prefix):]
        new_state_dict[new_key] = value

    if isinstance(base_model, LinearHead):
        # base_model.net.load_state_dict(new_state_dict, strict=False)
        load_state_dict_unstrict(base_model.net, new_state_dict)
    else:
        # base_model.load_state_dict(new_state_dict, strict=False)
        load_state_dict_unstrict(base_model, new_state_dict)


def load_state_dict_unstrict(base_model: nn.Module, state_dict: dict):
    msg = base_model.load_state_dict(state_dict, strict=False)
    _logger.info('missing keys: %s', msg.missing_keys)


def state_dict_to_cpu(state_dict: Dict[str, Tensor]) -> dict:
    state_dict_cpu = {k: v.clone().cpu() for k, v in state_dict.items()}
    return state_dict_cpu


def sanity_check(state_dict: Dict[str, Tensor], initial_state_dict: Dict[str, Tensor]):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    _logger.info('=> start sanity check')
    num_changes = 0
    for key in state_dict.keys():
        v1 = state_dict[key]
        v2 = initial_state_dict[key]
        is_same = torch.allclose(v1, v2)
        if not is_same:
            num_changes += 1
            _logger.warning('%s is not same', key)

    if num_changes <= 2:
        _logger.info('sanity check pass, num_changes=%d', num_changes)
    else:
        raise Exception
