"""
Error functions (mainly for use in notebooks).
"""

import torch


def calc_mse(output: torch.Tensor, target_output: torch.Tensor, dim: int = None) -> torch.Tensor:
    return ((output - target_output)**2).mean(dim=dim)


def calc_mse_rel(output: torch.Tensor, target_output: torch.Tensor, dim: int = None) -> torch.Tensor:
    return (
        ((output - target_output)**2).mean(dim=dim)
        / (target_output**2).mean(dim=dim)
    )


def calc_mae(output: torch.Tensor, target_output: torch.Tensor, dim: int = None) -> torch.Tensor:
    return (torch.abs(output - target_output)).mean(dim=dim)


def calc_mae_rel(output: torch.Tensor, target_output: torch.Tensor, dim: int = None) -> torch.Tensor:
    return (
        (torch.abs(output - target_output)).mean(dim=dim)
        / torch.abs(target_output).mean(dim=dim)
    )
