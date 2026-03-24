from collections.abc import Callable

import torch
from torch import nn

from src.models import ModalSystem
from src.utils.force import NonlinearStringSpectralForce, NonlinearStringTensorForce
from src.utils.func import ZeroFunc

FORCES: dict[str, Callable | nn.Module] = {
    "tensor": NonlinearStringTensorForce,
    "spectral": NonlinearStringSpectralForce,
    "zero": ZeroFunc
}


class NonlinearString(ModalSystem):
    def __init__(
            self,
            fs: int,
            dur: float,
            method_name: str,
            num_modes: int,
            nl_name: str = "spectral",
            method_kwargs: dict = {},
            nl_kwargs: dict = {}
        ):
        nl = FORCES[nl_name](num_modes=num_modes, **nl_kwargs)
        super().__init__(
            method_name,
            nl,
            **method_kwargs
        )

        self.register_buffer(
            "_fs",
            torch.as_tensor([int(fs)]),
            persistent=False
        )
        self.register_buffer(
            "_dur",
            torch.as_tensor([dur]),
            persistent=False
        )

        self._num_modes = num_modes
        self._omega = None
        self._sigma = None

    @staticmethod
    def _calc_modes(
            gamma: torch.Tensor,
            kappa: torch.Tensor,
            sigma0: torch.Tensor,
            sigma1: torch.Tensor,
            num_modes: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
        beta = torch.arange(start=1, end=(num_modes + 1), step=1, device=gamma.device) * torch.pi
        omega = torch.sqrt(torch.outer(gamma**2, beta**2) + torch.outer(kappa**2, beta**4))
        sigma = sigma0.unsqueeze(-1) + torch.outer(sigma1, beta**2)
        return omega, sigma

    @torch.inference_mode()
    def forward(
            self,
            y0: torch.Tensor,
            gamma: torch.Tensor,
            kappa: torch.Tensor,
            mu: torch.Tensor,
            sigma0: torch.Tensor,
            sigma1: torch.Tensor,
            xe: torch.Tensor,
            xo: torch.Tensor,
            exc_amp: torch.Tensor = None,
            exc_dur: torch.Tensor = None,
            exc_st: torch.Tensor = None,
            exc_type: torch.Tensor = None
        ):
        self._omega, self._sigma = self._calc_modes(gamma, kappa, sigma0, sigma1, self._num_modes)
        return super().forward(
            self._fs,
            self._dur,
            y0,
            self._omega,
            self._sigma,
            mu,
            xe,
            xo,
            exc_amp=exc_amp,
            exc_dur=exc_dur,
            exc_st=exc_st,
            exc_type=exc_type
        )

    @property
    def fs(self) -> int:
        return self._fs.item()

    @property
    def dur(self) -> float:
        return self._dur.item()

    @property
    def num_modes(self) -> int:
        return self._num_modes

    @property
    def omega(self) -> torch.Tensor:
        return self._omega

    @property
    def sigma(self) -> torch.Tensor:
        return self._sigma
