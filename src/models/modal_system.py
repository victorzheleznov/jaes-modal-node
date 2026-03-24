from collections.abc import Callable
from math import floor, sqrt

import torch
from torch import nn

from src.ode import ADJOINTS, IS_INTERLEAVED, METHODS, InitialValueProblem
from src.utils.excitation import Excitation


class ModalSystem(nn.Module):
    """Class for physical or physics-informed modelling using modal synthesis."""
    def __init__(
            self,
            method_name: str,
            nl: Callable | nn.Module,
            adjoint_name: str = "auto_diff",
            use_last: bool = False,
            use_jit: bool = False,
            **method_kwargs
        ):
        """Parameters
        ----------
        method_name : str
            Numerical method (as defined in `METHODS` dictionary).
        nl : Callable | nn.Module
            Nonlinear function that describes coupling between the modes (can be either a target nonlinearity or 
            a neural network used for training).
        adjoint_name : str
            Adjoint type for backpropogation (as defined in `ADJOINTS` dictionary).
        use_last : bool
            Flag to use only the last sample (True) or the whole trajectory (False) in the training loss.
        use_jit : bool
            Flag to use JIT compilation for the numerical solver.
        method_kwargs : dict
            Keyword arguments for the numerical method.
        """
        super().__init__()
        method = METHODS[method_name](nl=nl, **method_kwargs)
        self._is_interleaved = IS_INTERLEAVED[method_name]
        self._solver = ADJOINTS[adjoint_name](method=method, use_last=use_last)
        if use_jit:
            self._solver = torch.jit.script(self._solver)
        self._fe = Excitation()

    @staticmethod
    def _calc_num_samples(fs: torch.Tensor, dur: torch.Tensor) -> tuple[int, int]:
        assert torch.all(fs == fs[0]).item()
        assert torch.all(dur == dur[0]).item()
        fs = int(fs[0].item())
        dur = dur[0].item()
        num_samples = floor(dur * fs)
        return fs, num_samples

    @staticmethod
    def _Phi(x: torch.Tensor, num_modes: int) -> torch.Tensor:
        beta = torch.arange(start=1, end=(num_modes + 1), step=1, device=x.device) * torch.pi
        return sqrt(2) * torch.sin(torch.outer(x, beta))

    def _calc_exc(
            self,
            fs: int,
            num_samples: int,
            exc_amp: torch.Tensor,
            exc_dur: torch.Tensor,
            exc_st: torch.Tensor,
            exc_type: torch.Tensor,
            device: torch.device = None
        ) -> torch.Tensor:
        t_points = torch.arange(
            start=(0.5 if self._is_interleaved else 0),
            end=num_samples,
            step=1,
            device=device
        ) / fs
        self._fe.amp = exc_amp
        self._fe.dur = exc_dur
        self._fe.st = exc_st
        self._fe.type = exc_type
        return self._fe(t_points)

    def forward(
            self,
            fs: torch.Tensor,
            dur: torch.Tensor,
            y0: torch.Tensor,
            omega: torch.Tensor,
            sigma: torch.Tensor,
            mu: torch.Tensor,
            xe: torch.Tensor,
            xo: torch.Tensor,
            exc_amp: torch.Tensor = None,
            exc_dur: torch.Tensor = None,
            exc_st: torch.Tensor = None,
            exc_type: torch.Tensor = None,
            **batch
        ):
        """Synthesise model output for given simulation, modal and excitation parameters.

        Parameters
        ----------
        fs : torch.Tensor
            Sampling rate [Hz] (batch size,). Should be the same for the whole batch.
        dur : torch.Tensor
            Simulation duration [sec] (batch size,). Should be the same for the whole batch.
        y0 : torch.Tensor
            Initial conditions (batch size, state dimension).
        omega : torch.Tensor
            Modal angular frequencies (batch size, number of modes).
        sigma : torch.Tensor
            Modal damping parameters (batch size, number of modes).
        mu : torch.Tensor
            Scaling factors of the nonlinearity (batch size,). Denoted as "nu" in the JAES paper.
        xe : torch.Tensor
            Excitation positions (batch size,).
        xo : torch.Tensor
            Output positions (batch size,).
        exc_amp : torch.Tensor
            Excitation amplitudes (batch size,).
        exc_dur : torch.Tensor
            Excitation durations (batch size,).
        exc_st : torch.Tensor
            Excitation starting times (batch size,).
        exc_type : torch.Tensor
            Excitation types (batch size,).

        Returns
        -------
        out : dict[str, torch.Tensor]
            Computed system state ("output" key) and audio output ("w" key).
        """
        if not self.training:
            orig_use_last = self.use_last
            self.use_last = False

        fs, num_samples = self._calc_num_samples(fs, dur)
        num_modes = omega.shape[-1]
        Phi_e = self._Phi(xe, num_modes)
        fe_points = self._calc_exc(fs, num_samples, exc_amp, exc_dur, exc_st, exc_type, device=y0.device)
        y = self._solver(
            InitialValueProblem(
                fs,
                num_samples,
                y0,
                omega,
                sigma,
                mu,
                Phi_e,
                fe_points
            )
        )

        Phi_o = self._Phi(xo, num_modes)
        if not self.use_last:
            Phi_o = Phi_o.unsqueeze(-1)
        w = (Phi_o * y[:, :num_modes]).sum(1)

        if not self.training:
            self.use_last = orig_use_last

        return {"output": y, "w": w}

    @property
    def method_name(self):
        return str(self._solver._method)

    @property
    def nl(self):
        return self._solver._method._nl

    @property
    def use_last(self):
        return self._solver._use_last

    @use_last.setter
    def use_last(self, value: bool):
        if value is not None:
            self._solver._use_last = value
