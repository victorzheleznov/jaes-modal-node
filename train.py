import os
from datetime import timedelta

import hydra
import matplotlib
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from src.utils.data import get_dataloaders
from src.utils.init import set_random_seed, setup_logging, setup_saving
from src.utils.io import ROOT_PATH
from src.utils.misc import get_dtype
from src.utils.trainer import Trainer

matplotlib.use('Agg', force=False)

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
MASTER_RANK = 0


@hydra.main(version_base=None, config_path="cfg", config_name="nonlinear_oscillator")
def main(cfg):
    set_random_seed(cfg.seed)
    save_dir = ROOT_PATH / cfg.save_dir
    save_dir.mkdir(exist_ok=True, parents=True)
    setup_logging(save_dir, name=f"out_{LOCAL_RANK}")
    if LOCAL_RANK == MASTER_RANK:
        with open_dict(cfg):
            cfg.world_size = WORLD_SIZE
        setup_saving(cfg, save_dir)
        run_cfg = OmegaConf.to_container(cfg, resolve=True)
        writer = instantiate(cfg.writer, run_cfg)
        print(f"Training on {WORLD_SIZE:d} GPUs")
        print(f"Loaded {save_dir}")
    else:
        writer = None

    float_dtype = get_dtype(cfg.float_dtype)
    device = f"cuda:{LOCAL_RANK}"

    dataloaders = get_dataloaders(cfg)
    loss_func = instantiate(cfg.loss_func).to(float_dtype).to(device)
    model = instantiate(cfg.model).to(float_dtype).to(device)
    model = DDP(model, device_ids=[LOCAL_RANK])
    optimiser = instantiate(cfg.optimiser, params=model.parameters())
    metrics_func = instantiate(cfg.metrics_func)

    trainer = Trainer(
        model,
        loss_func,
        optimiser,
        device,
        dataloaders,
        cfg.monitor,
        save_dir,
        cfg.early_stop,
        cfg.log_step,
        cfg.max_grad_norm,
        metrics_func,
        writer,
        float_dtype
    )
    trainer.train(cfg.num_epochs)

    if LOCAL_RANK == MASTER_RANK:
        print(f"Results saved to {save_dir}")


if __name__ == "__main__":
    init_process_group(backend="nccl", timeout=timedelta(hours=10))
    main()
    destroy_process_group()
