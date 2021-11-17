from typing import List
from torch import nn, Tensor
import torch
from torchvision.models.resnet import ResNet
import torch.nn.functional as F


def create_projection(input_size: int, hidden_size: int, output_size: int):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, output_size)
    )


class Mbyol(nn.Module):

    def __init__(
        self,
        online_network: ResNet,
        target_network: ResNet,
        hidden_size: int = 2048,
        output_size: int = 128,
        m: float = 0.99,
    ):
        super().__init__()

        input_size = online_network.fc.in_features

        online_network.fc = nn.Identity()
        target_network.fc = nn.Identity()

        self.online_network = online_network
        self.target_network = target_network
        self.online_projection = create_projection(
            input_size, hidden_size, output_size,
        )
        self.target_projection = create_projection(
            input_size, hidden_size, output_size,
        )
        self.online_predictor = create_projection(
            output_size, hidden_size, output_size,
        )
        self.m = m

        self.momentum_update(0)

    def forward(self, view1: Tensor, view2: Tensor):
        q1 = self.online_network(view1)
        q1 = self.online_projection(q1)
        q1 = self.online_predictor(q1)

        with torch.no_grad():
            self.momentum_update(self.m)
            k2: Tensor = self.target_network(view2)
            k2: Tensor = self.target_projection(k2)
            k2 = (k2 > 0.).float()

        return q1, k2

    @staticmethod
    def _momentum_update(source: nn.Module, target: nn.Module, m: float):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data = m * target_param.data + \
                (1.0 - m) * source_param.data

    @torch.no_grad()
    def momentum_update(self, m: float):
        self._momentum_update(self.online_network, self.target_network, m)
        self._momentum_update(self.online_projection,
                              self.target_projection, m)
