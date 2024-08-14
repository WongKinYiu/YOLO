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
from typing import Any, Dict, List, Optional, Tuple, Union

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
from torch.nn import ModuleList
from torch.optim import Optimizer
from torchvision.transforms.functional import pil_to_tensor

from yolo.config.config import Config, YOLOLayer
from yolo.model.yolo import YOLO
from yolo.tools.drawer import draw_bboxes
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
        set_seed(cfg.lucky_number)
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.quite_mode = self.local_rank or getattr(cfg, "quite", False)
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
        if self.use_wandb and self.local_rank == 0:
            wandb.errors.term._log = custom_wandb_log
            self.wandb = wandb.init(
                project="YOLO", resume="allow", mode="online", dir=self.save_path, id=None, name=exp_name
            )

        self.use_tensorboard = cfg.use_tensorboard
        if self.use_tensorboard and self.local_rank == 0:
            from torch.utils.tensorboard import SummaryWriter

            self.tb_writer = SummaryWriter(log_dir=self.save_path / "tensorboard")
            logger.opt(colors=True).info(f"📍 Enable TensorBoard locally at <blue><u>http://localhost:6006</></>")

    def rank_check(logging_function):
        def wrapper(self, *args, **kwargs):
            if getattr(self, "local_rank", 0) != 0:
                return
            return logging_function(self, *args, **kwargs)

        return wrapper

    def get_renderable(self):
        renderable = Group(*self.get_renderables(), self.ap_table)
        return renderable

    @rank_check
    def start_train(self, num_epochs: int):
        self.task_epoch = self.add_task(f"[cyan]Start Training {num_epochs} epochs", total=num_epochs)
        self.update(self.task_epoch, advance=-0.5)

    @rank_check
    def start_one_epoch(
        self, num_batches: int, task: str = "Train", optimizer: Optimizer = None, epoch_idx: int = None
    ):
        self.num_batches = num_batches
        self.task = task
        if hasattr(self, "task_epoch"):
            self.update(self.task_epoch, description=f"[cyan] Preparing Data")

        if optimizer is not None:
            lr_values = [params["lr"] for params in optimizer.param_groups]
            lr_names = ["Learning Rate/bias", "Learning Rate/norm", "Learning Rate/conv"]
            if self.use_wandb:
                for lr_name, lr_value in zip(lr_names, lr_values):
                    self.wandb.log({lr_name: lr_value}, step=epoch_idx)

            if self.use_tensorboard:
                for lr_name, lr_value in zip(lr_names, lr_values):
                    self.tb_writer.add_scalar(lr_name, lr_value, global_step=epoch_idx)

        self.batch_task = self.add_task(f"[green] Phase: {task}", total=num_batches)

    @rank_check
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

    @rank_check
    def finish_one_epoch(self, batch_info: Dict[str, Any] = None, epoch_idx: int = -1):
        if self.task == "Train":
            prefix = "Loss"
        elif self.task == "Validate":
            prefix = "Metrics"
        batch_info = {f"{prefix}/{key}": value for key, value in batch_info.items()}
        if self.use_wandb:
            self.wandb.log(batch_info, step=epoch_idx)
        if self.use_tensorboard:
            for key, value in batch_info.items():
                self.tb_writer.add_scalar(key, value, epoch_idx)

        self.remove_task(self.batch_task)

    @rank_check
    def visualize_image(
        self,
        images: Optional[Tensor] = None,
        ground_truth: Optional[Tensor] = None,
        prediction: Optional[Union[List[Tensor], Tensor]] = None,
        epoch_idx: int = 0,
    ) -> None:
        """
        Upload the ground truth bounding boxes, predicted bounding boxes, and the original image to wandb or TensorBoard.

        Args:
            images (Optional[Tensor]): Tensor of images with shape (BZ, 3, 640, 640).
            ground_truth (Optional[Tensor]): Ground truth bounding boxes with shape (BZ, N, 5) or (N, 5). Defaults to None.
            prediction (prediction: Optional[Union[List[Tensor], Tensor]]): List of predicted bounding boxes with shape (N, 6) or (N, 6). Defaults to None.
            epoch_idx (int): Current epoch index. Defaults to 0.
        """
        if images is not None:
            images = images[0] if images.ndim == 4 else images
            if self.use_wandb:
                wandb.log({"Input Image": wandb.Image(images)}, step=epoch_idx)
            if self.use_tensorboard:
                self.tb_writer.add_image("Media/Input Image", images, 1)

        if ground_truth is not None:
            gt_boxes = ground_truth[0] if ground_truth.ndim == 3 else ground_truth
            if self.use_wandb:
                wandb.log(
                    {"Ground Truth": wandb.Image(images, boxes={"predictions": {"box_data": log_bbox(gt_boxes)}})},
                    step=epoch_idx,
                )
            if self.use_tensorboard:
                self.tb_writer.add_image("Media/Ground Truth", pil_to_tensor(draw_bboxes(images, gt_boxes)), epoch_idx)

        if prediction is not None:
            pred_boxes = prediction[0] if isinstance(prediction, list) else prediction
            if self.use_wandb:
                wandb.log(
                    {"Prediction": wandb.Image(images, boxes={"predictions": {"box_data": log_bbox(pred_boxes)}})},
                    step=epoch_idx,
                )
            if self.use_tensorboard:
                self.tb_writer.add_image("Media/Prediction", pil_to_tensor(draw_bboxes(images, pred_boxes)), epoch_idx)

    @rank_check
    def start_pycocotools(self):
        self.batch_task = self.add_task("[green]Run pycocotools", total=1)

    @rank_check
    def finish_pycocotools(self, result, epoch_idx=-1):
        ap_table, ap_main = make_ap_table(result * 100, self.ap_past_list, self.last_result, epoch_idx)
        self.last_result = np.maximum(result, self.last_result)
        self.ap_past_list.append((epoch_idx, ap_main))
        self.ap_table = ap_table

        if self.use_wandb:
            self.wandb.log({"PyCOCO/AP @ .5:.95": ap_main[2], "PyCOCO/AP @ .5": ap_main[5]})
        if self.use_tensorboard:
            # TODO: waiting torch bugs fix, https://github.com/pytorch/pytorch/issues/32651
            self.tb_writer.add_scalar("PyCOCO/AP @ .5:.95", ap_main[2], epoch_idx)
            self.tb_writer.add_scalar("PyCOCO/AP @ .5", ap_main[5], epoch_idx)

        self.update(self.batch_task, advance=1)
        self.refresh()
        self.remove_task(self.batch_task)

    @rank_check
    def finish_train(self):
        self.remove_task(self.task_epoch)
        self.stop()
        if self.use_wandb:
            self.wandb.finish()
        if self.use_tensorboard:
            self.tb_writer.close()


def custom_wandb_log(string="", level=int, newline=True, repeat=True, prefix=True, silent=False):
    if silent:
        return
    for line in string.split("\n"):
        logger.opt(raw=not newline, colors=True).info("🌐 " + line)


def log_model_structure(model: Union[ModuleList, YOLOLayer, YOLO]):
    if isinstance(model, YOLO):
        model = model.model
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


def log_bbox(
    bboxes: Tensor, class_list: Optional[List[str]] = None, image_size: Tuple[int, int] = (640, 640)
) -> List[dict]:
    """
    Convert bounding boxes tensor to a list of dictionaries for logging, normalized by the image size.

    Args:
        bboxes (Tensor): Bounding boxes with shape (N, 5) or (N, 6), where each box is [class_id, x_min, y_min, x_max, y_max, (confidence)].
        class_list (Optional[List[str]]): List of class names. Defaults to None.
        image_size (Tuple[int, int]): The size of the image, used for normalization. Defaults to (640, 640).

    Returns:
        List[dict]: List of dictionaries containing normalized bounding box information.
    """
    bbox_list = []
    scale_tensor = torch.Tensor([1, *image_size, *image_size]).to(bboxes.device)
    normalized_bboxes = bboxes[:, :5] / scale_tensor
    for bbox in normalized_bboxes:
        class_id, x_min, y_min, x_max, y_max, *conf = [float(val) for val in bbox]
        if class_id == -1:
            break
        bbox_entry = {
            "position": {"minX": x_min, "maxX": x_max, "minY": y_min, "maxY": y_max},
            "class_id": int(class_id),
        }
        if class_list:
            bbox_entry["box_caption"] = class_list[int(class_id)]
        if conf:
            bbox_entry["scores"] = {"confidence": conf[0]}
        bbox_list.append(bbox_entry)

    return bbox_list
