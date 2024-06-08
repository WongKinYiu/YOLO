import os

import torch
from loguru import logger
from torch import Tensor

# TODO: We may can't use CUDA?
from torch.cuda.amp import GradScaler, autocast

from yolo.config.config import Config, TrainConfig, ValidationConfig
from yolo.model.yolo import YOLO
from yolo.tools.data_loader import StreamDataLoader, create_dataloader
from yolo.tools.drawer import draw_bboxes
from yolo.tools.loss_functions import create_loss_function
from yolo.utils.bounding_box_utils import Vec2Box, bbox_nms, calculate_map
from yolo.utils.logging_utils import ProgressTracker
from yolo.utils.model_utils import (
    ExponentialMovingAverage,
    create_optimizer,
    create_scheduler,
)


class ModelTrainer:
    def __init__(self, cfg: Config, model: YOLO, vec2box: Vec2Box, save_path: str, device):
        train_cfg: TrainConfig = cfg.task
        self.model = model
        self.vec2box = vec2box
        self.device = device
        self.optimizer = create_optimizer(model, train_cfg.optimizer)
        self.scheduler = create_scheduler(self.optimizer, train_cfg.scheduler)
        self.loss_fn = create_loss_function(cfg, vec2box)
        self.progress = ProgressTracker(cfg.name, save_path, cfg.use_wandb)
        self.num_epochs = cfg.task.epoch

        self.validation_dataloader = create_dataloader(cfg.task.validation.data, cfg.dataset, cfg.task.validation.task)
        self.validator = ModelValidator(cfg.task.validation, model, vec2box, save_path, device, self.progress)

        if getattr(train_cfg.ema, "enabled", False):
            self.ema = ExponentialMovingAverage(model, decay=train_cfg.ema.decay)
        else:
            self.ema = None
        self.scaler = GradScaler()

    def train_one_batch(self, images: Tensor, targets: Tensor):
        images, targets = images.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()

        with autocast():
            predicts = self.model(images)
            aux_predicts = self.vec2box(predicts["AUX"])
            main_predicts = self.vec2box(predicts["Main"])
            loss, loss_item = self.loss_fn(aux_predicts, main_predicts, targets)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item(), loss_item

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        for images, targets in dataloader:
            loss, loss_each = self.train_one_batch(images, targets)

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

    def solve(self, dataloader):
        logger.info("🚄 Start Training!")
        num_epochs = self.num_epochs

        with self.progress.progress:
            self.progress.start_train(num_epochs)
            for epoch in range(num_epochs):

                self.progress.start_one_epoch(len(dataloader), self.optimizer, epoch)
                epoch_loss = self.train_one_epoch(dataloader)
                self.progress.finish_one_epoch()

                self.validator.solve(self.validation_dataloader)


class ModelTester:
    def __init__(self, cfg: Config, model: YOLO, vec2box: Vec2Box, save_path: str, device):
        self.model = model
        self.device = device
        self.vec2box = vec2box
        self.progress = ProgressTracker(cfg, save_path, cfg.use_wandb)

        self.nms = cfg.task.nms
        self.save_path = save_path
        self.save_predict = getattr(cfg.task, "save_predict", None)
        self.idx2label = cfg.class_list

    def solve(self, dataloader: StreamDataLoader):
        logger.info("👀 Start Inference!")
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()

        if dataloader.is_stream:
            import cv2
            import numpy as np
        try:
            for idx, images in enumerate(dataloader):
                images = images.to(self.device)
                with torch.no_grad():
                    predicts = self.model(images)
                    predicts = self.vec2box(predicts["Main"])
                nms_out = bbox_nms(predicts[0], predicts[2], self.nms)
                img = draw_bboxes(images[0], nms_out[0], idx2label=self.idx2label)

                if dataloader.is_stream:
                    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    cv2.imshow("Prediction", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    if not self.save_predict:
                        continue

                if self.save_predict == False:
                    save_image_path = os.path.join(self.save_path, f"frame{idx:03d}.png")
                    img.save(save_image_path)
                    logger.info(f"💾 Saved visualize image at {save_image_path}")

        except (KeyboardInterrupt, Exception) as e:
            dataloader.stop_event.set()
            dataloader.stop()
            if isinstance(e, KeyboardInterrupt):
                logger.error("User Keyboard Interrupt")
            else:
                raise e
        dataloader.stop()


class ModelValidator:
    def __init__(
        self,
        validation_cfg: ValidationConfig,
        model: YOLO,
        vec2box: Vec2Box,
        save_path: str,
        device,
        # TODO: think Progress?
        progress: ProgressTracker,
    ):
        self.model = model
        self.vec2box = vec2box
        self.device = device
        self.progress = progress
        self.save_path = save_path

        self.nms = validation_cfg.nms

    def solve(self, dataloader):
        # logger.info("🧪 Start Validation!")
        self.model.eval()
        # TODO: choice mAP metrics?
        iou_thresholds = torch.arange(0.5, 1.0, 0.05)
        map_all = []
        self.progress.start_one_epoch(len(dataloader))
        for images, targets in dataloader:
            images, targets = images.to(self.device), targets.to(self.device)
            with torch.no_grad():
                predicts = self.model(images)
            predicts = self.vec2box(predicts["Main"])
            nms_out = bbox_nms(predicts[0], predicts[2], self.nms)
            for idx, predict in enumerate(nms_out):
                map_value = calculate_map(predict, targets[idx], iou_thresholds)
                map_all.append(map_value[0])
            self.progress.one_batch(mapp=torch.Tensor(map_all).mean())

        self.progress.finish_one_epoch()
