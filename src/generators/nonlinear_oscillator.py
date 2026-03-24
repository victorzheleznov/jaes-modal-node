from collections.abc import Callable

import torch

from src.models import ModalSystem
from src.utils.func import PowFunc, SinhFunc, TanhFunc, ZeroFunc

FUNCTIONS: dict[str, Callable] = {
    "pow": PowFunc,
    "tanh": TanhFunc,
    "sinh": SinhFunc,
    "zero": ZeroFunc
}


class NonlinearOscillator(ModalSystem):
    def __init__(
            self,
            fs: int,
            dur: float,
            method_name: str,
            nl_name: str = "pow",
            method_kwargs: dict = {},
            nl_kwargs: dict = {}
        ):
        nl = FUNCTIONS[nl_name](**nl_kwargs)
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

        self.register_buffer(
            "_xe",
            torch.as_tensor([0.25]),
            persistent=False
        )
        self.register_buffer(
            "_xo",
            torch.as_tensor([0.25]),
            persistent=False
        )

    @torch.inference_mode()
    def forward(
            self,
            y0: torch.Tensor,
            omega: torch.Tensor,
            sigma: torch.Tensor,
            mu: torch.Tensor,
            exc_amp: torch.Tensor = None,
            exc_dur: torch.Tensor = None,
            exc_st: torch.Tensor = None,
            exc_type: torch.Tensor = None
        ):
        return super().forward(
            self._fs,
            self._dur,
            y0,
            omega,
            sigma,
            mu,
            self._xe,
            self._xo,
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
    def xe(self) -> float:
        return self._xe.item()

    @property
    def xo(self) -> float:
        return self._xo.item()
