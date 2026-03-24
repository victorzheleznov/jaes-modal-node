from collections.abc import Callable
from typing import NamedTuple, Tuple

import torch
from torch import nn

from ..problem import InitialValueProblem


class EulerParameters(NamedTuple):
    k: float
    omega_sq: torch.Tensor
    mu_sq: torch.Tensor
    sigma: torch.Tensor
    Phi_e: torch.Tensor
    fe_points: torch.Tensor


class Euler(nn.Module):
    def __init__(self, nl: Callable | nn.Module, **kwargs):
        super().__init__()
        self._nl = nl

    def __str__(self):
        return "euler"

    @torch.jit.export
    def init(self, ivp: InitialValueProblem) -> Tuple[torch.Tensor, EulerParameters]:
        # precompute parameters
        k = 1.0 / ivp.fs
        omega_sq = ivp.omega**2
        mu_sq = (ivp.mu**2).unsqueeze(-1)
        return (
            ivp.y0,
            EulerParameters(k, omega_sq, mu_sq, ivp.sigma, ivp.Phi_e, ivp.fe_points)
        )

    @torch.jit.export
    def step(
            self,
            n: int,
            y0: torch.Tensor,
            params: EulerParameters
        ) -> torch.Tensor:
        # parse input
        q0, p0 = torch.chunk(y0, 2, dim=-1)
        k, omega_sq, mu_sq, sigma, Phi_e, fe_points = params

        # update state
        q1 = q0 + k * p0
        nl0 = self._nl(q0)
        fe0 = fe_points[:, n].unsqueeze(-1)
        p1 = p0 + k * (-2.0 * sigma * p0 - omega_sq * q0 + mu_sq * nl0 + Phi_e * fe0)
        y1 = torch.cat([q1, p1], dim=-1)

        return y1