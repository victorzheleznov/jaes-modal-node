import os
from contextlib import redirect_stdout

from hydra.utils import instantiate
from torch.distributed import barrier
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
MASTER_RANK = 0


def get_dataloaders(cfg):
    collate_func = instantiate(cfg.collate_func)

    dataloaders = dict()
    for partition in cfg.datasets.keys():
        if LOCAL_RANK == MASTER_RANK:
            dataset = instantiate(cfg.datasets[partition])
            barrier(device_ids=[LOCAL_RANK])
        else:
            barrier(device_ids=[LOCAL_RANK])
            with redirect_stdout(None):
                dataset = instantiate(cfg.datasets[partition])
        dataloader = DataLoader(
            dataset,
            cfg.batch_size,
            shuffle=False,
            collate_fn=collate_func,
            sampler=DistributedSampler(dataset, shuffle=(partition == "train"))
        )
        dataloaders[partition] = dataloader

    return dataloaders
