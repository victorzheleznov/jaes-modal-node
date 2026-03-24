import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


def read_json(file: Path | str) -> dict:
    """Read the .json file."""
    file = Path(file)
    with open(file, "r") as f:
        return json.load(f)


def write_json(file: Path | str, data: dict):
    """Write dictionary to a .json file"""
    file = Path(file)
    with open(file, "w") as f:
        json.dump(data, f, indent=4, sort_keys=False)


def read_wav(file: Path | str) -> tuple[torch.Tensor, int]:
    """Read the audio .wav file"""
    file = Path(file)
    data, fs = sf.read(file)
    data = torch.from_numpy(data)
    return data, fs


def write_wav(
        file: Path | str,
        data: np.ndarray | torch.Tensor,
        fs: int,
        normalise: bool = False,
        subtype: str = "DOUBLE"
    ):
    """Write audio to a .wav file"""
    file = Path(file)
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if normalise:
        data = data / np.max(np.abs(data))
    sf.write(file, data, samplerate=int(fs), subtype=subtype)
