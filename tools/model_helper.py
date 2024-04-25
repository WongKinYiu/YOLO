from typing import Any, Dict, Type

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from config.config import OptimizerConfig, SchedulerConfig


class EMA:
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


def get_optimizer(model_parameters, optim_cfg: OptimizerConfig) -> Optimizer:
    """Create an optimizer for the given model parameters based on the configuration.

    Returns:
        An instance of the optimizer configured according to the provided settings.
    """
    optimizer_class: Type[Optimizer] = getattr(torch.optim, optim_cfg.type)
    return optimizer_class(model_parameters, **optim_cfg.args)


def get_scheduler(optimizer: Optimizer, schedul_cfg: SchedulerConfig) -> _LRScheduler:
    """Create a learning rate scheduler for the given optimizer based on the configuration.

    Returns:
        An instance of the scheduler configured according to the provided settings.
    """
    scheduler_class: Type[_LRScheduler] = getattr(torch.optim.lr_scheduler, schedul_cfg.type)
    return scheduler_class(optimizer, **schedul_cfg.args)
