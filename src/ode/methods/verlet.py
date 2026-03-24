from collections.abc import Callable
from typing import NamedTuple, Tuple

import torch
from torch import nn

from ..problem import InitialValueProblem


class VerletParameters(NamedTuple):
    k: float
    dp: torch.Tensor
    dm: torch.Tensor
    omega_sq: torch.Tensor
    mu_sq: torch.Tensor
    Phi_e: torch.Tensor
    fe_points: torch.Tensor


class Verlet(nn.Module):
    def __init__(self, nl: Callable | nn.Module, **kwargs):
        super().__init__()
        self._nl = nl

    def __str__(self):
        return "verlet"

    @torch.jit.export
    def init(self, ivp: InitialValueProblem) -> Tuple[torch.Tensor, VerletParameters]:
        # precompute parameters
        k = 1.0 / ivp.fs
        dp = 1.0 + k * ivp.sigma
        dm = 1.0 - k * ivp.sigma
        omega_sq = ivp.omega**2
        mu_sq = (ivp.mu**2).unsqueeze(-1)
        return (
            ivp.y0,
            VerletParameters(k, dp, dm, omega_sq, mu_sq, ivp.Phi_e, ivp.fe_points)
        )

    @torch.jit.export
    def step(
            self,
            n: int,
            y0: torch.Tensor,
            params: VerletParameters
        ) -> torch.Tensor:
        # parse input
        q0, p0 = torch.chunk(y0, 2, dim=-1)
        k, dp, dm, omega_sq, mu_sq, Phi_e, fe_points = params

        # update state
        q_half = q0 + 0.5 * k * p0
        nl_half = self._nl(q_half)
        fe_half = fe_points[:, n].unsqueeze(-1)
        p1 = (dm * p0 + k * (-omega_sq * q_half + mu_sq * nl_half + Phi_e * fe_half)) / dp
        q1 = q_half + 0.5 * k * p1
        y1 = torch.cat([q1, p1], dim=-1)

        return y1
