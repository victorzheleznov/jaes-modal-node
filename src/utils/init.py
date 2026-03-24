import logging
import os
import random
import subprocess
from functools import partial
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv, set_key
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn

from src.utils.io import ROOT_PATH
from wandb.util import generate_id


def set_random_seed(seed: int):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    update_env("PYTHON_SEED", str(seed))


def setup_saving(cfg: DictConfig, save_dir: Path):
    cfg_file = save_dir / "cfg.yaml"
    if cfg_file.is_file():
        run_id = (OmegaConf.load(cfg_file)).writer.run_id
    else:
        run_id = generate_id()
    with open_dict(cfg):
        cfg.writer.run_id = run_id
    log_git(save_dir)
    OmegaConf.save(cfg, cfg_file, resolve=True)
    update_env("PYTHON_LAST_SAVE_DIR", str(save_dir))
    return save_dir


def setup_logging(save_dir: Path, name: str = "out"):
    log_file = save_dir / f"{name}.log"
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(module)s: %(message)s",
        force=True
    )


def log_git(save_dir: Path):
    commit_file = save_dir / "git_commit.txt"
    patch_file = save_dir / "git_diff.patch"
    with commit_file.open("w") as f:
        subprocess.call(["git", "rev-parse", "HEAD"], stdout=f)
    with patch_file.open("w") as f:
        subprocess.call(["git", "diff", "HEAD"], stdout=f)


def update_env(key: str, value: str):
    os.environ[key] = value
    dotenv_file = ROOT_PATH / ".env"
    dotenv_file.touch(exist_ok=True)
    set_key(dotenv_file, key, value)


def load_env():
    dotenv_file = ROOT_PATH / ".env"
    if dotenv_file.exists():
        load_dotenv(dotenv_file)


def get_init_func(activation: nn.Module, name: str, **kwargs):
    if "xavier" in name:
        if "LeakyReLU" in str(activation):
            kwargs.update({"gain": nn.init.calculate_gain("leaky_relu", param=activation.negative_slope)})
        elif "ReLU" in str(activation):
            kwargs.update({"gain": nn.init.calculate_gain("relu")})
        elif "Tanh" in str(activation):
            kwargs.update({"gain": nn.init.calculate_gain("tanh")})
        elif "Sigmoid" in str(activation):
            kwargs.update({"gain": nn.init.calculate_gain("sigmoid")})
    elif "kaiming" in name:
        if "LeakyReLU" in str(activation):
            kwargs.update({"nonlinearity": "leaky_relu", "a": activation.negative_slope})
        elif "ReLU" in str(activation):
            kwargs.update({"nonlinearity": "relu"})

    func = getattr(nn.init, name)
    func = partial(func, **kwargs)

    return func
