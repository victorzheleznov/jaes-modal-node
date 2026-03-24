"""
Custom activation functions (which include `antiderivative` and `potential` methods).

N.B. Definition of potential (without a minus!):
$f(x) = \nabla_x V(x)$
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class ReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)

    def antiderivative(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.square(F.relu(x))

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        return self.antiderivative(x).sum(-1)


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 1e-2):
        super().__init__()
        self._negative_slope = negative_slope

    def extra_repr(self) -> str:
        return f"negative_slope={self._negative_slope}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, negative_slope=self._negative_slope)

    def antiderivative(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.square(
            F.leaky_relu(x, negative_slope=math.sqrt(self._negative_slope))
        )

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        return self.antiderivative(x).sum(-1)

    @property
    def negative_slope(self):
        return self._negative_slope


class PReLU(nn.Module):
    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super().__init__()
        self._num_parameters = num_parameters
        self._weight = nn.Parameter(torch.empty(self._num_parameters), requires_grad=True)
        torch.nn.init.constant_(self._weight, init)

    def extra_repr(self) -> str:
        return f"num_parameters={self._num_parameters}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.prelu(x, weight=self._weight)

    def antiderivative(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.square(
            F.prelu(x, weight=torch.sqrt(self._weight))
        )

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        return self.antiderivative(x).sum(-1)


class Tanh(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.tanh(x)

    def antiderivative(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.cosh(x))

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        return self.antiderivative(x).sum(-1)


class Sigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.sigmoid(x)

    def antiderivative(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x, beta=1.0)

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        return self.antiderivative(x).sum(-1)


class Exp(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)

    def antiderivative(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        return self.antiderivative(x).sum(-1)


class Pow(nn.Module):
    def __init__(self, exponent: float = 2):
        super().__init__()
        self._exponent  = exponent

    def extra_repr(self) -> str:
        return f"exponent={self._exponent}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.pow(x, self._exponent)

    def antiderivative(self, x: torch.Tensor) -> torch.Tensor:
        return torch.pow(x, self._exponent + 1) / (self._exponent + 1)

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        return self.antiderivative(x).sum(-1)

    @property
    def exponent(self):
        return self._exponent


class Linear(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def antiderivative(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.square(x)

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        return self.antiderivative(x).sum(-1)


class Heaviside(nn.Module):
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.heaviside(x, torch.as_tensor(0.0, device=x.device, dtype=x.dtype))

    def antiderivative(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        return self.antiderivative(x).sum(-1)


class Softmax(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=-1)

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(x, dim=-1)
