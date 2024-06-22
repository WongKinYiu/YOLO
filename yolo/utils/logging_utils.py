"""
Module for initializing logging tools used in machine learning and data processing.
Supports integration with Weights & Biases (wandb), Loguru, TensorBoard, and other
logging frameworks as needed.

This setup ensures consistent logging across various platforms, facilitating
effective monitoring and debugging.

Example:
    from tools.logger import custom_logger
    custom_logger()
"""

import os
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import wandb
import wandb.errors.term
from loguru import logger
from omegaconf import ListConfig
from rich.console import Console, Group
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch import Tensor
from torch.optim import Optimizer

from yolo.config.config import Config, YOLOLayer
from yolo.utils.solver_utils import make_ap_table


def custom_logger(quite: bool = False):
    logger.remove()
    if quite:
        return
    logger.add(
        sys.stderr,
        colorize=True,
        format="<fg #003385>[{time:MM/DD HH:mm:ss}]</> <level>{level: ^8}</level>| <level>{message}</level>",
    )


# TODO: should be moved to correct position
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ProgressLogger(Progress):
    def __init__(self, cfg: Config, exp_name: str, *args, **kwargs):
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.quite_mode = local_rank or getattr(cfg, "quite", False)
        custom_logger(self.quite_mode)
        self.save_path = validate_log_directory(cfg, exp_name=cfg.name)

        progress_bar = (
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("{task.completed:.0f}/{task.total:.0f}"),
            TimeRemainingColumn(),
        )
        self.ap_table = Table()
        # TODO: load maxlen by config files
        self.ap_past_list = deque(maxlen=5)
        self.last_result = 0
        super().__init__(*args, *progress_bar, **kwargs)

        self.use_wandb = cfg.use_wandb
        if self.use_wandb:
            wandb.errors.term._log = custom_wandb_log
            self.wandb = wandb.init(
                project="YOLO", resume="allow", mode="online", dir=self.save_path, id=None, name=exp_name
            )

    def get_renderable(self):
        renderable = Group(*self.get_renderables(), self.ap_table)
        return renderable

    def start_train(self, num_epochs: int):
        self.task_epoch = self.add_task(f"[cyan]Start Training {num_epochs} epochs", total=num_epochs)

    def start_one_epoch(
        self, num_batches: int, task: str = "Train", optimizer: Optimizer = None, epoch_idx: int = None
    ):
        self.num_batches = num_batches
        self.task = task
        if hasattr(self, "task_epoch"):
            self.update(self.task_epoch, description=f"[cyan] Preparing Data")

        if self.use_wandb and optimizer is not None:
            lr_values = [params["lr"] for params in optimizer.param_groups]
            lr_names = ["Learning Rate/bias", "Learning Rate/norm", "Learning Rate/conv"]
            for lr_name, lr_value in zip(lr_names, lr_values):
                self.wandb.log({lr_name: lr_value}, step=epoch_idx)
        self.batch_task = self.add_task(f"[green] Phase: {task}", total=num_batches)

    def one_batch(self, batch_info: Dict[str, Tensor] = None):
        epoch_descript = "[cyan]" + self.task + "[white] |"
        batch_descript = "|"
        if self.task == "Train":
            self.update(self.task_epoch, advance=1 / self.num_batches)
        for info_name, info_val in batch_info.items():
            epoch_descript += f"{info_name: ^9}|"
            batch_descript += f"   {info_val:2.2f}  |"
        self.update(self.batch_task, advance=1, description=f"[green]{self.task} [white]{batch_descript}")
        if hasattr(self, "task_epoch"):
            self.update(self.task_epoch, description=epoch_descript)

    def finish_one_epoch(self, batch_info: Dict[str, Any] = None, epoch_idx: int = -1):
        if self.task == "Train":
            prefix = "Loss/"
        elif self.task == "Validate":
            prefix = "Metrics/"
        batch_info = {f"{prefix}{key}": value for key, value in batch_info.items()}
        if self.use_wandb:
            self.wandb.log(batch_info, step=epoch_idx)
        self.remove_task(self.batch_task)

    def start_pycocotools(self):
        self.batch_task = self.add_task("[green]Run pycocotools", total=1)

    def finish_pycocotools(self, result, epoch_idx=-1):
        ap_table, ap_main = make_ap_table(result, self.ap_past_list, self.last_result, epoch_idx)
        self.last_result = np.maximum(result, self.last_result)
        self.ap_past_list.append((epoch_idx, ap_main))
        self.ap_table = ap_table

        if self.use_wandb:
            self.wandb.log({"PyCOCO/AP @ .5:.95": ap_main[2], "PyCOCO/AP @ .5": ap_main[5]})
        self.update(self.batch_task, advance=1)
        self.refresh()
        self.remove_task(self.batch_task)

    def finish_train(self):
        self.remove_task(self.task_epoch)
        self.stop()
        self.wandb.finish()


def custom_wandb_log(string="", level=int, newline=True, repeat=True, prefix=True, silent=False):
    if silent:
        return
    for line in string.split("\n"):
        logger.opt(raw=not newline, colors=True).info("🌐 " + line)


def log_model_structure(model: List[YOLOLayer]):
    console = Console()
    table = Table(title="Model Layers")

    table.add_column("Index", justify="center")
    table.add_column("Layer Type", justify="center")
    table.add_column("Tags", justify="center")
    table.add_column("Params", justify="right")
    table.add_column("Channels (IN->OUT)", justify="center")

    for idx, layer in enumerate(model, start=1):
        layer_param = sum(x.numel() for x in layer.parameters())  # number parameters
        in_channels, out_channels = getattr(layer, "in_c", None), getattr(layer, "out_c", None)
        if in_channels and out_channels:
            if isinstance(in_channels, (list, ListConfig)):
                in_channels = "M"
            if isinstance(out_channels, (list, ListConfig)):
                out_channels = "M"
            channels = f"{str(in_channels): >4} -> {str(out_channels): >4}"
        else:
            channels = "-"
        table.add_row(str(idx), layer.layer_type, layer.tags, f"{layer_param:,}", channels)
    console.print(table)


def validate_log_directory(cfg: Config, exp_name: str) -> Path:
    base_path = Path(cfg.out_path, cfg.task.task)
    save_path = base_path / exp_name

    if not cfg.exist_ok:
        index = 1
        old_exp_name = exp_name
        while save_path.is_dir():
            exp_name = f"{old_exp_name}{index}"
            save_path = base_path / exp_name
            index += 1
        if index > 1:
            logger.opt(colors=True).warning(
                f"🔀 Experiment directory exists! Changed <red>{old_exp_name}</> to <green>{exp_name}</>"
            )

    save_path.mkdir(parents=True, exist_ok=True)
    logger.opt(colors=True).info(f"📄 Created log folder: <u><fg #808080>{save_path}</></>")
    logger.add(save_path / "output.log", mode="w", backtrace=True, diagnose=True)
    return save_path
