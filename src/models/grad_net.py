"""
Implementation of Gradient Networks [1].

N.B. Only architectures that allow for a non-negative and closed-form potential are considered.

References
----------
[1] S. Chaudhari, S. Pranav and J. M. F. Moura, "Gradient Networks", IEEE Trans. Signal Process., vol. 73, pp. 324-339,
    2025.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.utils.init import get_init_func


class MGradNetModule(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            activation: nn.Module,
            transform: nn.Module | None = None,
            bias: bool = True
        ):
        super().__init__()
        assert hasattr(activation, "potential")
        assert callable(activation.potential)
        if transform is not None:
            assert hasattr(transform, "antiderivative")
            assert callable(transform.antiderivative)
        self._weight = nn.Parameter(torch.empty((hidden_dim, in_dim)), requires_grad=True)
        self.register_parameter(
            "_bias",
            nn.Parameter(torch.empty((hidden_dim,)), requires_grad=True)
            if bias
            else None
        )
        self._activation = activation
        self._transform = transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # linear layer
        x = F.linear(x, self._weight, bias=self._bias)

        # activation
        z = self._activation(x)
        z = F.linear(z, self._weight.T)

       # transform
        if self._transform is not None:
            y = self._activation.potential(x)
            y = self._transform(y)
            z = z * y.unsqueeze(-1)

        return torch.neg(z)

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        # linear layer
        x = F.linear(x, self._weight, bias=self._bias)

        # activation
        y = self._activation.potential(x)

        # transform
        if self._transform is not None:
            y = self._transform.antiderivative(y)

        return y

    @torch.jit.export
    def forward_and_potential(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate force and potential with one shared affine projection."""
        # linear layer
        x = F.linear(x, self._weight, bias=self._bias)

        # activation
        z = self._activation(x)
        z = F.linear(z, self._weight.T)
        y = self._activation.potential(x)

        # transform
        if self._transform is not None:
            z = z * self._transform(y).unsqueeze(-1)
            y = self._transform.antiderivative(y)

        return torch.neg(z), y


class BaseMGradNet(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: list[int],
            activation: nn.Module,
            transform: nn.Module | None = None,
            bias: bool = True,
            log_coefs_std: float = 1,
            init_name: str = "xavier_uniform_",
            **init_kwargs
        ):
        super().__init__()
        self._list = nn.ModuleList(
            [MGradNetModule(in_dim, n, activation, transform, bias) for n in hidden_dim]
        )
        self._log_coefs = nn.Parameter(torch.empty((len(self._list),)), requires_grad=True)
        self._log_coefs_std = log_coefs_std
        self._init_func = get_init_func(activation, init_name, **init_kwargs)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, MGradNetModule):
            self._init_func(module._weight)
            if module._bias is not None:
                nn.init.zeros_(module._bias)
        if isinstance(module, BaseMGradNet):
            nn.init.normal_(module._log_coefs, std=self._log_coefs_std)

    @torch.jit.export
    def potential(self, x: torch.Tensor) -> torch.Tensor:
        y = 0
        for i, module in enumerate(self._list):
            y += self._log_coefs[i].exp() * module.potential(x)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = 0
        for i, module in enumerate(self._list):
            y += self._log_coefs[i].exp() * module(x)
        return y

    @torch.jit.export
    def forward_and_potential(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate both outputs without recomputing each module's projection."""
        force = 0
        potential = 0
        for i, module in enumerate(self._list):
            module_force, module_potential = module.forward_and_potential(x)
            coef = self._log_coefs[i].exp()
            force += coef * module_force
            potential += coef * module_potential
        return force, potential


class BaseCGradNet(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            activation: nn.Module,
            bias: bool = True,
            log_scale_std: float = 1,
            init_name: str = "xavier_uniform_",
            **init_kwargs
        ):
        super().__init__()
        assert hasattr(activation, "antiderivative")
        assert callable(activation.antiderivative)
        self._weight = nn.Parameter(torch.empty((hidden_dim, in_dim)), requires_grad=True)
        self.register_parameter(
            "_bias",
            nn.Parameter(torch.empty((hidden_dim,)), requires_grad=True)
            if bias
            else None
        )
        # scaling vectors are always positive
        self._log_scale_in = nn.Parameter(torch.empty((hidden_dim,)), requires_grad=True)
        self._log_scale_out = nn.Parameter(torch.empty((hidden_dim,)), requires_grad=True)
        self._activation = activation
        self._log_scale_std = log_scale_std
        self._init_func = get_init_func(activation, init_name, **init_kwargs)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, BaseCGradNet):
            self._init_func(module._weight)
            nn.init.normal_(module._log_scale_in, std=self._log_scale_std)
            nn.init.normal_(module._log_scale_out, std=self._log_scale_std)
            if module._bias is not None:
                nn.init.zeros_(module._bias)

    @torch.jit.export
    def potential(self, x: torch.Tensor) -> torch.Tensor:
        x = F.linear(x, self._log_scale_in.exp().unsqueeze(-1) * self._weight, bias=self._bias)
        y = self._activation.antiderivative(x)
        y = self._log_scale_out.exp().unsqueeze(0) * y
        return y.sum(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.linear(x, self._log_scale_in.exp().unsqueeze(-1) * self._weight, bias=self._bias)
        y = self._activation(x)
        y = (self._log_scale_out.exp() * self._log_scale_in.exp()).unsqueeze(0) * y
        y = F.linear(y, self._weight.T)
        return torch.neg(y)

    @torch.jit.export
    def forward_and_potential(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate force and potential with one shared affine projection."""
        scale_in = self._log_scale_in.exp()
        scale_out = self._log_scale_out.exp()
        x = F.linear(x, scale_in.unsqueeze(-1) * self._weight, bias=self._bias)

        y = self._activation(x)
        y = (scale_out * scale_in).unsqueeze(0) * y
        y = F.linear(y, self._weight.T)

        potential = self._activation.antiderivative(x)
        potential = (scale_out.unsqueeze(0) * potential).sum(-1)
        return torch.neg(y), potential
