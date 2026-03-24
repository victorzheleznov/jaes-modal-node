"""
Nonlinear forces for modal systems.

N.B. Definition of potential:
$f(q) = -\nabla_q V(q)$
"""

import math
from functools import partial

import torch
from torch import nn
from torch_dct import LinearDCT, dct, idct

from src.utils.func import NonlinearStringExactFunc, PowFunc


class NonlinearStringTensorForce(nn.Module):
    def __init__(self, num_modes: int):
        super().__init__()
        self.register_buffer(
            "_tensor",
            self._calc_tensor(num_modes),
            persistent=False
        )
        self._tensor = self._tensor.flatten(start_dim=1).to_sparse_csr()

    @staticmethod
    def _calc_tensor(num_modes: int) -> torch.Tensor:
        def delta(n: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            return torch.eq(n, m).int()

        v = torch.arange(start=1, end=(num_modes + 1), step=1)
        m, i, j, k = torch.meshgrid(v, v, v, v, indexing="ij")

        B = i * j * k**2 * (
            delta(k + i, m + j) - delta(k + i, -(m + j))
            + delta(k + i, m - j) - delta(k + i, -(m - j))
            + delta(k - i, m + j) - delta(k - i, -(m + j))
            + delta(k - i, m - j) - delta(k - i, -(m - j))
        )

        # full tensor
        A = 0.5 * torch.pi**4 * (
            B
            + torch.permute(B, (0, 3, 1, 2))
            + torch.permute(B, (0, 2, 3, 1))
        )

        # simplified tensor (using symmetry properties)
        #A = 1.5 * torch.pi**4 * B

        return A

    def _grad_potential(self, q: torch.Tensor) -> torch.Tensor:
        q_vec = torch.einsum("bi,bj,bk->ijkb", q, q, q).flatten(end_dim=-2)
        out = torch.matmul(self._tensor, q_vec).T
        return out

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return torch.neg(self._grad_potential(q))

    def potential(self, q: torch.Tensor) -> torch.Tensor:
        grad_potential = self._grad_potential(q)
        out = 0.25 * (grad_potential * q).sum(-1)
        return out


class NonlinearStringSpectralForce(nn.Module):
    def __init__(self, num_modes: int, use_exact: bool = True, use_linear: bool = False):
        super().__init__()
        self._num_modes = num_modes
        self.register_buffer(
            "_beta",
            (torch.arange(start=1, end=(self._num_modes + 1), step=1) * torch.pi).unsqueeze(0),
            persistent=False
        )
        if use_exact:
            self._func = NonlinearStringExactFunc()
        else:
            # the same cubic approximation as in `NonlinearStringTensorForce`
            self._func = PowFunc(factor=1.0, exponent=3)
        assert hasattr(self._func, "antiderivative")
        assert callable(self._func.antiderivative)
        if use_linear:
            self._dct = LinearDCT(in_features=(self._num_modes + 1), type="dct", norm="ortho", bias=False)
            self._idct = LinearDCT(in_features=(self._num_modes + 1), type="idct", norm="ortho", bias=False)
        else:
            self._dct = partial(dct, norm="ortho")
            self._idct = partial(idct, norm="ortho")

    def _calc_xi(self, q: torch.Tensor) -> torch.Tensor:
        xi = self._idct(
            torch.cat(
                [
                    torch.zeros((q.shape[0], 1), device=q.device),
                    self._beta * q
                ],
                dim=-1
            )
        ) * math.sqrt(self._num_modes + 1)
        return xi

    def _calc_f_q(self, f_xi: torch.Tensor) -> torch.Tensor:
        f_q = torch.neg(
            self._beta * (self._dct(f_xi)[..., 1:]) / math.sqrt(self._num_modes + 1)
        )
        return f_q

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        xi = self._calc_xi(q)
        f_xi = self._func(xi)
        out = self._calc_f_q(f_xi)
        return out

    def potential(self, q: torch.Tensor) -> torch.Tensor:
        xi = self._calc_xi(q)
        out = self._func.antiderivative(xi).sum(-1) / (self._num_modes + 1)
        return out
