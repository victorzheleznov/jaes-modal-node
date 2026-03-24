import torch

from src.metrics import BaseMetric


class MSE(BaseMetric):
    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        output, target_output = self._parse_batch(batch)
        mse = ((output - target_output)**2).mean()
        return {str(self): mse}


class MSEPosVel(BaseMetric):
    def __init__(self, per_mode: bool = False):
        super().__init__("output")
        self._per_mode = per_mode

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        output, target_output = self._parse_batch(batch)
        num_modes = batch["omega"].shape[-1]
        q, p, _ = torch.tensor_split(output, (num_modes, 2 * num_modes), dim=1)
        target_q, target_p, _ = torch.tensor_split(target_output, (num_modes, 2 * num_modes), dim=1)
        mse_pos = ((q - target_q)**2).mean()
        mse_vel = ((p - target_p)**2).mean()
        d = {"mse_pos": mse_pos, "mse_vel": mse_vel}
        if self._per_mode:
            mse_pos_per_mode = ((q - target_q)**2).mean(dim=[0, -1])
            mse_vel_per_mode = ((p - target_p)**2).mean(dim=[0, -1])
            d.update({f"mse_pos_{(m+1):d}": mse_pos_per_mode[m] for m in range(num_modes)})
            d.update({f"mse_vel_{(m+1):d}": mse_vel_per_mode[m] for m in range(num_modes)})
        return d


class MSERel(BaseMetric):
    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        output, target_output = self._parse_batch(batch)
        mse_rel = (
            ((output - target_output)**2).mean()
            / (target_output**2).mean()
        )
        return {str(self): mse_rel}
