from collections.abc import Callable
from typing import NamedTuple, Tuple

import torch
from torch import nn

from ..problem import InitialValueProblem


class SAVParameters(NamedTuple):
    k: float
    dp: torch.Tensor
    dm: torch.Tensor
    omega_sq: torch.Tensor
    mu: torch.Tensor
    Phi_e: torch.Tensor
    fe_points: torch.Tensor


class SAV(nn.Module):
    def __init__(self, nl: Callable | nn.Module, eps: float = 1e-12, lambda0: float = 0, **kwargs):
        super().__init__()
        assert hasattr(nl, "potential")
        assert callable(nl.potential)
        self._nl = nl
        self._eps = eps
        self._lambda0 = lambda0

    def __str__(self):
        return "sav"

    @torch.jit.export
    def init(self, ivp: InitialValueProblem) -> Tuple[torch.Tensor, SAVParameters]:
        # precompute parameters
        k = 1.0 / ivp.fs
        dp = 1.0 + k * ivp.sigma
        dm = 1.0 - k * ivp.sigma
        omega_sq = ivp.omega**2
        mu = ivp.mu.unsqueeze(-1)

        # calculate initial condition
        num_modes = ivp.omega.shape[-1]
        if ivp.y0.shape[-1] == (2 * num_modes):
            q0, p0 = torch.chunk(ivp.y0, 2, dim=-1)
            potential0 = self._nl.potential(q0)
            psi0 = self._calc_psi(potential0)
            y0 = torch.cat([q0, p0, psi0], dim=-1)
        else:
            y0 = ivp.y0
        return (
            y0,
            SAVParameters(k, dp, dm, omega_sq, mu, ivp.Phi_e, ivp.fe_points)
        )

    @torch.jit.export
    def step(
            self,
            n: int,
            y0: torch.Tensor,
            params: SAVParameters
        ) -> torch.Tensor:
        # parse input
        k, dp, dm, omega_sq, mu, Phi_e, fe_points = params
        num_modes = omega_sq.shape[-1]
        q0, p0, psi0 = torch.tensor_split(y0, (num_modes, 2 * num_modes), dim=-1)

        # update state
        q_half = q0 + 0.5 * k * p0
        g0 = self._calc_g_std(q_half) + self._calc_g_mod(q0, p0, psi0)
        a0 = 0.5 * k * mu * g0
        b0 = a0 / dp
        fe_half = fe_points[:, n].unsqueeze(-1)
        p1 = (
            dm * p0
            - a0 * (a0 * p0).sum(-1, keepdim=True)
            + k * (-omega_sq * q_half - mu**2 * g0 * psi0 + Phi_e * fe_half)
        )
        p1 = (
            p1 / dp
            - (b0 * (b0 * p1).sum(-1, keepdim=True) / (1.0 + (a0 * b0).sum(-1, keepdim=True)))
        )
        p_half = 0.5 * (p0 + p1)
        psi1 = psi0 + k * (g0 * p_half).sum(-1, keepdim=True)
        q1 = q_half + 0.5 * k * p1
        y1 = torch.cat([q1, p1, psi1], dim=-1)

        return y1

    @torch.jit.export
    def _calc_psi(self, potential: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(2.0 * potential + self._eps).unsqueeze(-1)

    @torch.jit.export
    def _calc_g_std(self, q: torch.Tensor) -> torch.Tensor:
        nl = self._nl(q)
        potential = self._nl.potential(q)
        true_psi = self._calc_psi(potential)
        g = -nl / true_psi
        return g

    @torch.jit.ignore
    @torch.no_grad()
    def _calc_g_mod(self, q: torch.Tensor, p: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        if self._lambda0:
            potential = self._nl.potential(q)
            true_psi = self._calc_psi(potential)
            norm_p = (torch.sign(p) * p).sum(-1, keepdim=True)
            g = -self._lambda0 * (psi - true_psi) * torch.sign(p) / (norm_p + self._eps)
        else:
            g = torch.zeros_like(p)
        return g
