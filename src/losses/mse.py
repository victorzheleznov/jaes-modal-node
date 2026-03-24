import torch
from torch import nn

from src.losses import BaseLoss


class MSELoss(BaseLoss):
    def __init__(self, tensor_name: str = "output", **kwargs):
        super().__init__(tensor_name=tensor_name, **kwargs)
        self._loss = nn.MSELoss()

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        output, target_output = self._process_batch(batch)
        mse_loss = self._loss(output, target_output)
        return {"loss": mse_loss, "mse_loss": mse_loss}


class MSEPosVelLoss(BaseLoss):
    def __init__(self, w_pos: float = 1.0, w_vel: float = 1.0, w_psi: float = 0.0, **kwargs):
        super().__init__(tensor_name="output", **kwargs)
        self._w_pos = w_pos
        self._w_vel = w_vel
        self._w_psi = w_psi
        self._loss = nn.MSELoss()

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        output, target_output = self._process_batch(batch)
        num_modes = batch["omega"].shape[-1]
        q, p, psi = torch.tensor_split(output, (num_modes, 2 * num_modes), dim=1)
        target_q, target_p, target_psi = torch.tensor_split(target_output, (num_modes, 2 * num_modes), dim=1)

        mse_pos_loss = self._apply_weight(self._loss, q, target_q, self._w_pos)
        mse_vel_loss = self._apply_weight(self._loss, p, target_p, self._w_vel)
        loss = mse_pos_loss + mse_vel_loss
        d = {"mse_pos_loss": mse_pos_loss, "mse_vel_loss": mse_vel_loss}

        if (target_psi.shape[1] > 0) and (psi.shape[1] > 0):
            mse_psi_loss = self._apply_weight(self._loss, psi, target_psi, self._w_psi)
            loss += mse_psi_loss
            d.update({"mse_psi_loss": mse_psi_loss})

        d.update({"loss": loss})
        return d
