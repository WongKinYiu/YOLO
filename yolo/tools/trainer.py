import torch
from loguru import logger
from tqdm import tqdm

from yolo.config.config import TrainConfig
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

    def train_one_batch(self, data, targets):
        data, targets = data.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(data)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()
        if self.ema:
            self.ema.update()
        return loss.item()

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for data, targets in tqdm(dataloader, desc="Training"):
            loss = self.train_one_batch(data, targets)
            total_loss += loss
        if self.scheduler:
            self.scheduler.step()
        return total_loss / len(dataloader)

    def save_checkpoint(self, epoch, filename="checkpoint.pt"):
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
