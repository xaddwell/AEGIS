import torch
from torch import nn


def get_submodule_weight(module: nn.Module, layer_name: str) -> torch.Tensor:
    return module.get_submodule(layer_name).weight.detach().clone()


@torch.no_grad()
def set_submodule_weight(module: nn.Module, layer_name: str, weight: torch.Tensor):
    module.get_submodule(layer_name).weight.copy_(weight)
