import copy

import torch
from torch import nn

from src.utils.init import get_init_func


class BaseMLP(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: list[int],
            out_dim: int,
            activation: nn.Module,
            bias: bool = True,
            init_name: str = "xavier_uniform_",
            **init_kwargs
        ):
        super().__init__()
        self._net = nn.Sequential()
        for n in hidden_dim:
            self._net.append(nn.Linear(in_features=in_dim, out_features=n, bias=bias))
            self._net.append(copy.deepcopy(activation))
            in_dim = n
        self._net.append(nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias))
        self._init_func = get_init_func(activation, init_name, **init_kwargs)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            self._init_func(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)
