import gc
import hashlib

import matplotlib.pyplot as plt
import scipy
import torch
from PIL import Image


def calc_md5(s: str):
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def get_image(fig: plt.Figure):
    fig.canvas.draw()
    img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)
    gc.collect()
    return img


def get_window(name: str, len: int, sym: bool = False) -> torch.Tensor:
    try:
        win = getattr(torch.signal.windows, name)(len, sym=sym)
    except AttributeError:
        win = getattr(scipy.signal.windows, name)(len, sym=sym)
        win = torch.from_numpy(win)
    return win


def gen_from_range(
        range: tuple[float, float],
        shape: tuple[int, ...],
        dtype: torch.dtype = None,
        device: torch.device = None,
        generator: torch.Generator = None
    ) -> torch.Tensor:
    if range[0] == range[1]:
        out = range[0] * torch.ones(shape, dtype=dtype, device=device)
    else:
        out = torch.rand(shape, dtype=dtype, device=device, generator=generator)
        out = range[0] + (range[1] - range[0]) * out
    return out


def get_dtype(dtype: str = None) -> torch.dtype:
    if dtype is not None:
        dtype = getattr(torch, dtype)
        assert isinstance(dtype, torch.dtype)
    return dtype


def move_batch_to_device(batch: dict, device: torch.device, float_dtype: torch.dtype = None):
    for key in batch.keys():
        if type(batch[key]) is torch.Tensor:
            batch[key] = batch[key].to(device)
            if torch.is_floating_point(batch[key]) and float_dtype is not None:
                batch[key] = batch[key].to(float_dtype)
    return batch
