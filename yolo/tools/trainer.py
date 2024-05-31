import torch
from loguru import logger
from torch import Tensor

# TODO: We may can't use CUDA?
from torch.cuda.amp import GradScaler, autocast

from yolo.config.config import Config, TrainConfig
from yolo.model.yolo import get_model
from yolo.tools.log_helper import CustomProgress
from yolo.tools.model_helper import EMA, get_optimizer, get_scheduler
from yolo.utils.loss import get_loss_function


class Trainer:
    def __init__(self, cfg: Config, save_path: str, device):
        train_cfg: TrainConfig = cfg.hyper.train
        model = get_model(cfg)

        self.model = model.to(device)
        self.device = device
        self.optimizer = get_optimizer(model, train_cfg.optimizer)
        self.scheduler = get_scheduler(self.optimizer, train_cfg.scheduler)
        self.loss_fn = get_loss_function(cfg)
        self.progress = CustomProgress(cfg, save_path, use_wandb=True)

        if getattr(train_cfg.ema, "enabled", False):
            self.ema = EMA(model, decay=train_cfg.ema.decay)
        else:
            self.ema = None
        self.scaler = GradScaler()

    def train_one_batch(self, data: Tensor, targets: Tensor):
        data, targets = data.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()

        with autocast():
            outputs = self.model(data)
            loss, loss_item = self.loss_fn(outputs, targets)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item(), loss_item

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        for data, targets in dataloader:
            loss, loss_each = self.train_one_batch(data, targets)

            total_loss += loss
            self.progress.one_batch(loss_each)

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
        logger.info("🚄 Start Training!")

        with self.progress.progress:
            self.progress.start_train(num_epochs)
            for epoch in range(num_epochs):

                self.progress.start_one_epoch(len(dataloader), self.optimizer, epoch)
                epoch_loss = self.train_one_epoch(dataloader)
                self.progress.finish_one_epoch()

                logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
                if (epoch + 1) % 5 == 0:
                    self.save_checkpoint(epoch, f"checkpoint_epoch_{epoch+1}.pth")
