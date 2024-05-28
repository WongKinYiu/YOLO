import torch
from loguru import logger
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from yolo.config.config import Config, TrainConfig
from yolo.model.yolo import YOLO
from yolo.tools.model_helper import EMA, get_optimizer, get_scheduler
from yolo.utils.loss import get_loss_function


class Trainer:
    def __init__(self, model: YOLO, cfg: Config, device):
        train_cfg: TrainConfig = cfg.hyper.train

        self.model = model.to(device)
        self.device = device
        self.optimizer = get_optimizer(model.parameters(), train_cfg.optimizer)
        self.scheduler = get_scheduler(self.optimizer, train_cfg.scheduler)
        self.loss_fn = get_loss_function()

        if train_cfg.ema.get("enabled", False):
            self.ema = EMA(model, decay=train_cfg.ema.decay)
        else:
            self.ema = None
        self.scaler = GradScaler()

    def train_one_batch(self, data: Tensor, targets: Tensor, progress: tqdm):
        data, targets = data.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()

        with autocast():
            outputs = self.model(data)
            loss, loss_item = self.loss_fn(outputs, targets)
            loss_iou, loss_dfl, loss_cls = loss_item

        progress.set_description(f"Loss IoU: {loss_iou:.5f}, DFL: {loss_dfl:.5f}, CLS: {loss_cls:.5f}")

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.ema:
            self.ema.update()

        return loss.item()

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        with tqdm(dataloader, desc="Training") as progress:
            for data, targets in progress:
                loss = self.train_one_batch(data, targets, progress)
                total_loss += loss
            if self.scheduler:
                self.scheduler.step()
        return total_loss / len(dataloader)

    def save_checkpoint(self, epoch: int, filename="checkpoint.pt"):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.ema:
            self.ema.apply_shadow()
            checkpoint["model_state_dict_ema"] = self.model.state_dict()
            self.ema.restore()
        torch.save(checkpoint, filename)

    def train(self, dataloader, num_epochs):
        logger.info("start train")
        for epoch in range(num_epochs):
            epoch_loss = self.train_one_epoch(dataloader)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, f"checkpoint_epoch_{epoch+1}.pth")
