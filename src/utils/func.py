"""
Nonlinear functions (mainly for nonlinear oscillator simulations).

N.B. Definition of potential:
$f(x) = -\nabla_x V(x)$
"""

import torch


class PowFunc:
    def __init__(self, factor: float = -1.0, exponent: float = 3):
        self._factor = factor
        self._exponent = exponent

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._factor * torch.pow(x, self._exponent)

    def antiderivative(self, x: torch.Tensor) -> torch.Tensor:
        return self._factor / (self._exponent + 1) * torch.pow(x, self._exponent + 1)

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        return torch.neg(self.antiderivative(x).sum(-1))


class ZeroFunc:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def antiderivative(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)


class TanhFunc:
    def __init__(self, factor: float = -1.0):
        self._factor = factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._factor * torch.tanh(x)

    def antiderivative(self, x: torch.Tensor) -> torch.Tensor:
        return self._factor * torch.log(torch.cosh(x))

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        return torch.neg(self.antiderivative(x).sum(-1))


class SinhFunc:
    def __init__(self, factor: float = -1.0):
        self._factor = factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._factor * torch.sinh(x)

    def antiderivative(self, x: torch.Tensor) -> torch.Tensor:
        return self._factor * torch.cosh(x)

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        return torch.neg(self.antiderivative(x).sum(-1))


class NonlinearStringExactFunc:
    def __call__(self, xi: torch.Tensor) -> torch.Tensor:
        return 2.0 * xi * (1.0 - 1.0 / torch.sqrt(1.0 + torch.square(xi)))

    def antiderivative(self, xi: torch.Tensor) -> torch.Tensor:
        return torch.square(torch.sqrt(1.0 + torch.square(xi)) - 1.0)
