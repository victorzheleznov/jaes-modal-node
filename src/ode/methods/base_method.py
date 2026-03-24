from collections.abc import Callable
from typing import NamedTuple, Tuple

import torch
from torch import nn

from ..problem import InitialValueProblem


class BaseMethodParameters(NamedTuple):
    pass


class BaseMethod(nn.Module):
    """Template for numerical methods."""
    def __init__(self, nl: Callable | nn.Module, **kwargs):
        """Parameters
        ----------
        nl : Callable | nn.Module
            Nonlinear function that describes coupling between the modes (can be either a target nonlinearity or 
            a neural network used for training). Included in class initialisation to allow JIT compilation.
        kwargs : dict
            Keyword arguments for the numerical method.
        """
        raise NotImplementedError()

    def __str__(self):
        return "base_method"

    def init(self, ivp: InitialValueProblem) -> Tuple[torch.Tensor, BaseMethodParameters]:
        """Initialise parameters for the numerical method.
        
        Parameters
        ----------
        ivp : InitialValueProblem
            Initial value problem.

        Returns
        -------
        : torch.Tensor
            Updated initial conditions (batch size, state dimension). Useful for appending auxiliary variables.
        : BaseMethodParameters
            Parameters of the numerical method (as defined in `BaseMethodParameters` named tuple).
            Useful for precomputing vectors for the state update equation.
        """
        raise NotImplementedError()

    def step(
            self,
            n: int,
            y0: torch.Tensor,
            params: BaseMethodParameters
        ) -> torch.Tensor:
        """Calculate step of the numerical method.

        Parameters
        ----------
        n : int
            Current time index.
        y0 : torch.Tensor
            Current state (batch size, state dimension).
        params : BaseMethodParameters
            Parameters of the numerical method (as defined in `BaseMethodParameters` named tuple).

        Returns
        -------
        : torch.Tensor
            Next state (batch size, state dimension).
        """
        raise NotImplementedError()
