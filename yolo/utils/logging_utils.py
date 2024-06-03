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
import sys
from typing import Dict, List

import wandb
import wandb.errors.term
from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from torch import Tensor
from torch.optim import Optimizer

from yolo.config.config import Config, GeneralConfig, YOLOLayer


def custom_logger():
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<fg #003385>[{time:MM/DD HH:mm:ss}]</> <level>{level: ^8}</level>| <level>{message}</level>",
    )


class ProgressTracker:
    def __init__(self, cfg: Config, save_path: str, use_wandb: bool = False):
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("{task.completed:.0f}/{task.total:.0f}"),
            TimeRemainingColumn(),
        )
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.errors.term._log = custom_wandb_log
            self.wandb = wandb.init(
                project="YOLO", resume="allow", mode="online", dir=save_path, id=None, name=cfg.name
            )

    def start_train(self, num_epochs: int):
        self.task_epoch = self.progress.add_task("[cyan]Epochs  [white]| Loss | Box  | DFL  | BCE  |", total=num_epochs)

    def start_one_epoch(self, num_batches: int, optimizer: Optimizer, epoch_idx: int):
        self.num_batches = num_batches
        if self.use_wandb:
            lr_values = [params["lr"] for params in optimizer.param_groups]
            lr_names = ["bias", "norm", "conv"]
            for lr_name, lr_value in zip(lr_names, lr_values):
                self.wandb.log({f"Learning Rate/{lr_name}": lr_value}, step=epoch_idx)
        self.batch_task = self.progress.add_task("[green]Batches", total=num_batches)

    def one_batch(self, loss_dict: Dict[str, Tensor]):
        if self.use_wandb:
            for loss_name, loss_value in loss_dict.items():
                self.wandb.log({f"Loss/{loss_name}": loss_value})

        loss_str = "| -.-- |"
        for loss_name, loss_val in loss_dict.items():
            loss_str += f" {loss_val:2.2f} |"

        self.progress.update(self.batch_task, advance=1, description=f"[green]Batches [white]{loss_str}")
        self.progress.update(self.task_epoch, advance=1 / self.num_batches)

    def finish_one_epoch(self):
        self.progress.remove_task(self.batch_task)

    def finish_train(self):
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
            channels = f"{in_channels:4} -> {out_channels:4}"
        else:
            channels = "-"
        table.add_row(str(idx), layer.layer_type, layer.tags, f"{layer_param:,}", channels)
    console.print(table)


def validate_log_directory(general_cfg: GeneralConfig, exp_name):
    base_path = os.path.join(general_cfg.out_path, general_cfg.task)
    save_path = os.path.join(base_path, exp_name)

    if not general_cfg.exist_ok:
        index = 1
        old_exp_name = exp_name
        while os.path.isdir(save_path):
            exp_name = f"{old_exp_name}{index}"
            save_path = os.path.join(base_path, exp_name)
            index += 1
        if index > 1:
            logger.opt(colors=True).warning(
                f"🔀 Experiment directory exists! Changed <red>{old_exp_name}</> to <green>{exp_name}</>"
            )

    os.makedirs(save_path, exist_ok=True)
    logger.opt(colors=True).info(f"📄 Created log folder: <u><fg #808080>{save_path}</></>")
    logger.add(os.path.join(save_path, "output.log"), backtrace=True, diagnose=True)
    return save_path
