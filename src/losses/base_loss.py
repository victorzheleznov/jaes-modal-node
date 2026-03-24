from collections.abc import Callable

import torch
from torch import nn


class BaseLoss(nn.Module):
    def __init__(self, tensor_name: str = "output"):
        super().__init__()
        self._tensor_name = tensor_name

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def _process_batch(self, batch: dict) -> tuple[torch.Tensor]:
        output = batch[self._tensor_name]
        target_output = batch["target_" + self._tensor_name]

        if output.ndim == (target_output.ndim - 1):
            # account for `use_last` option in adjoints
            target_output = target_output[..., -1]

        assert output.shape == target_output.shape
        return output, target_output

    @staticmethod
    def _apply_weight(
            func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            output: torch.Tensor,
            target_output: torch.Tensor,
            weight: float
        ) -> torch.Tensor:
        loss = (
            weight * func(output, target_output)
            if weight
            else torch.as_tensor(0.0, dtype=target_output.dtype, device=target_output.device)
        )
        return loss
