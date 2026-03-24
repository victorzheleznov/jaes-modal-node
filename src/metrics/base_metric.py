import re

import torch
from torch import nn


class BaseMetric(nn.Module):
    def __init__(self, tensor_name: str = "output"):
        super().__init__()
        self._tensor_name = tensor_name
        self._metric_name = "_".join(
            re.sub("([A-Z][a-z]+)", r" \1",
                   re.sub("([A-Z]+)", r" \1", self.__class__.__name__)
            ).split()
        ).lower()

    def __str__(self):
        return f"{self._metric_name}_{self._tensor_name}"

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        output, target_output = self._parse_batch(batch)
        raise NotImplementedError

    def _parse_batch(self, batch: dict) -> tuple[torch.Tensor]:
        output = batch[self._tensor_name]
        target_output = batch["target_" + self._tensor_name]

        # account for `use_last` option in adjoints
        if output.ndim == (target_output.ndim - 1):
            target_output = target_output[..., -1]

        assert output.shape == target_output.shape
        return output, target_output


class MetricList(nn.Module):
    def __init__(self, metrics: list[BaseMetric]):
        super().__init__()
        self._metrics = metrics

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        d = dict()
        for metric in self._metrics:
            d.update(metric(batch))
        return d
