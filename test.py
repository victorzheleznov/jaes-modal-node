import os
from collections import defaultdict
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from tabulate import tabulate
from torch.distributed import ReduceOp, destroy_process_group, init_process_group, reduce
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from src.utils.data import get_dataloaders
from src.utils.init import load_env, set_random_seed
from src.utils.io import ROOT_PATH
from src.utils.misc import get_dtype, move_batch_to_device

load_env()

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
MASTER_RANK = 0


@hydra.main(version_base=None, config_path=os.environ["PYTHON_LAST_SAVE_DIR"], config_name="cfg")
def main(cfg):
    set_random_seed(cfg.seed)
    save_dir = Path([path["path"] for path in HydraConfig.get().runtime.config_sources if path["provider"] == "main"][0])
    if LOCAL_RANK == MASTER_RANK:
        print(f"Loaded {save_dir}")

    float_dtype = get_dtype(cfg.float_dtype)
    device = f"cuda:{LOCAL_RANK}"

    dataloaders = get_dataloaders(cfg)
    loss_func = instantiate(cfg.loss_func).to(float_dtype).to(device)
    model = instantiate(cfg.model).to(float_dtype).to(device)
    model.load_state_dict(torch.load(save_dir / "checkpoint_best.pt", weights_only=True)["model_state_dict"])
    model = DDP(model, device_ids=[LOCAL_RANK])
    metrics_func = instantiate(cfg.metrics_func)

    for partition, dataloader in dataloaders.items():
        dataset = dataloader.dataset
        if hasattr(dataset, "save_pred"):
            if callable(dataset.save_pred):
                test_dir = ROOT_PATH / "test" / save_dir.stem / str(dataset)
                test_dir.mkdir(exist_ok=True, parents=True)
                with open(test_dir / "params.txt", "w") as f:
                    f.write(f"{str(dataset.params)}\n{dataset.md5}")
                with open(test_dir / "partition.txt", "w") as f:
                    f.write(partition)

        with torch.no_grad():
            model.eval()
            running_losses = defaultdict(lambda: 0.0)
            running_metrics = defaultdict(lambda: 0.0)
            for batch in tqdm(dataloader, disable=(LOCAL_RANK != MASTER_RANK)):
                batch = move_batch_to_device(batch, device, float_dtype)
                pred = model(**batch)
                batch.update(pred)
                losses = loss_func(batch)
                for key in losses.keys():
                    running_losses[key] += losses[key]
                if metrics_func:
                    metrics = metrics_func(batch)
                    for key in metrics.keys():
                        running_metrics[key] += metrics[key]
                if hasattr(dataset, "save_pred"):
                    if callable(dataset.save_pred):
                        dataset.save_pred(test_dir, **batch)

            for key in running_losses.keys():
                reduce(running_losses[key], dst=MASTER_RANK, op=ReduceOp.SUM)
            if LOCAL_RANK == MASTER_RANK:
                len_dataloader = (len(dataloader) * WORLD_SIZE)
                average_losses = {
                    key: (running_losses[key].item() / len_dataloader)
                    for key in running_losses.keys()
                }
                loss = average_losses.pop("loss")
                print(f"Loss for {partition} dataset: {loss:.2e}\n")
                print(tabulate(average_losses.items(), headers=["Loss", "Value"], floatfmt=".2e") + "\n")

            if metrics_func:
                for key in running_metrics.keys():
                    reduce(running_metrics[key], dst=MASTER_RANK, op=ReduceOp.SUM)
                if LOCAL_RANK == MASTER_RANK:
                    average_metrics = {
                        key: (running_metrics[key].item() / len_dataloader)
                        for key in running_metrics.keys()
                    }
                    print(tabulate(average_metrics.items(), headers=["Metric", "Value"], floatfmt=".2e") + "\n")


if __name__ == "__main__":
    init_process_group(backend="nccl")
    main()
    destroy_process_group()
