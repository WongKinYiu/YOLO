import os
from typing import List, Type, Union

import torch
import torch.distributed as dist
from loguru import logger
from omegaconf import ListConfig
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, _LRScheduler

from yolo.config.config import OptimizerConfig, SchedulerConfig
from yolo.model.yolo import YOLO


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}

    def update(self):
        """Update the shadow parameters using the current model parameters."""
        for name, param in self.model.named_parameters():
            assert name in self.shadow, "All model parameters should have a corresponding shadow parameter."
            new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
            self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply the shadow parameters to the model."""
        for name, param in self.model.named_parameters():
            param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore the original parameters from the shadow."""
        for name, param in self.model.named_parameters():
            self.shadow[name].copy_(param.data)


def create_optimizer(model: YOLO, optim_cfg: OptimizerConfig) -> Optimizer:
    """Create an optimizer for the given model parameters based on the configuration.

    Returns:
        An instance of the optimizer configured according to the provided settings.
    """
    optimizer_class: Type[Optimizer] = getattr(torch.optim, optim_cfg.type)

    bias_params = [p for name, p in model.named_parameters() if "bias" in name]
    norm_params = [p for name, p in model.named_parameters() if "weight" in name and "bn" in name]
    conv_params = [p for name, p in model.named_parameters() if "weight" in name and "bn" not in name]

    model_parameters = [
        {"params": bias_params, "nestrov": True, "momentum": 0.937},
        {"params": conv_params, "weight_decay": 0.0},
        {"params": norm_params, "weight_decay": 1e-5},
    ]
    return optimizer_class(model_parameters, **optim_cfg.args)


def create_scheduler(optimizer: Optimizer, schedule_cfg: SchedulerConfig) -> _LRScheduler:
    """Create a learning rate scheduler for the given optimizer based on the configuration.

    Returns:
        An instance of the scheduler configured according to the provided settings.
    """
    scheduler_class: Type[_LRScheduler] = getattr(torch.optim.lr_scheduler, schedule_cfg.type)
    schedule = scheduler_class(optimizer, **schedule_cfg.args)
    if hasattr(schedule_cfg, "warmup"):
        wepoch = schedule_cfg.warmup.epochs
        lambda1 = lambda epoch: 0.1 + 0.9 * (epoch + 1 / wepoch) if epoch < wepoch else 1
        lambda2 = lambda epoch: 10 - 9 * (epoch / wepoch) if epoch < wepoch else 1
        warmup_schedule = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2, lambda1])
        schedule = SequentialLR(optimizer, schedulers=[warmup_schedule, schedule], milestones=[2])
    return schedule


def initialize_distributed() -> None:
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    logger.info(f"Initialized process group; rank: {rank}, size: {world_size}")
    return local_rank


def get_device(device_spec: Union[str, int, List[int]]) -> torch.device:
    ddp_flag = False
    if isinstance(device_spec, (list, ListConfig)):
        ddp_flag = True
        device_spec = initialize_distributed()
    device = torch.device(device_spec)
    return device, ddp_flag
