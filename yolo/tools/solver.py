import torch
from loguru import logger
from torch import Tensor

# TODO: We may can't use CUDA?
from torch.cuda.amp import GradScaler, autocast

from yolo.config.config import Config, TrainConfig, ValidationConfig
from yolo.model.yolo import YOLO
from yolo.tools.data_loader import StreamDataLoader, create_dataloader
from yolo.tools.drawer import draw_bboxes
from yolo.tools.loss_functions import get_loss_function
from yolo.utils.bounding_box_utils import AnchorBoxConverter, bbox_nms, calculate_map
from yolo.utils.logging_utils import ProgressTracker
from yolo.utils.model_utils import (
    ExponentialMovingAverage,
    create_optimizer,
    create_scheduler,
)


class ModelTrainer:
    def __init__(self, cfg: Config, model: YOLO, save_path: str, device):
        train_cfg: TrainConfig = cfg.task
        self.model = model
        self.device = device
        self.optimizer = create_optimizer(model, train_cfg.optimizer)
        self.scheduler = create_scheduler(self.optimizer, train_cfg.scheduler)
        self.loss_fn = get_loss_function(cfg)
        self.progress = ProgressTracker(cfg.name, save_path, cfg.use_wandb)
        self.num_epochs = cfg.task.epoch

        validation_dataloader = create_dataloader(cfg.task.validation.data, cfg.dataset, cfg.task.validation.task)
        anchor2box = AnchorBoxConverter(cfg.model, cfg.image_size, device)
        self.validator = ModelValidator(
            cfg.task.validation, model, save_path, device, self.progress, anchor2box, validation_dataloader
        )

        if getattr(train_cfg.ema, "enabled", False):
            self.ema = ExponentialMovingAverage(model, decay=train_cfg.ema.decay)
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

    def solve(self, dataloader):
        logger.info("🚄 Start Training!")
        num_epochs = self.num_epochs

        with self.progress.progress:
            self.progress.start_train(num_epochs)
            for epoch in range(num_epochs):

                self.progress.start_one_epoch(len(dataloader), self.optimizer, epoch)
                epoch_loss = self.train_one_epoch(dataloader)
                self.progress.finish_one_epoch()

                self.validator.solve()


class ModelTester:
    def __init__(self, cfg: Config, model: YOLO, save_path: str, device):
        self.model = model
        self.device = device
        self.progress = ProgressTracker(cfg, save_path, cfg.use_wandb)

        self.anchor2box = AnchorBoxConverter(cfg.model, cfg.image_size, device)
        self.nms = cfg.task.nms
        self.save_path = save_path

    def solve(self, dataloader: StreamDataLoader):
        logger.info("👀 Start Inference!")

        try:
            for idx, images in enumerate(dataloader):
                images = images.to(self.device)
                with torch.no_grad():
                    raw_output = self.model(images)
                predict, _ = self.anchor2box(raw_output[0][3:], with_logits=True)
                nms_out = bbox_nms(predict, self.nms)
                draw_bboxes(
                    images[0], nms_out[0], scaled_bbox=False, save_path=self.save_path, save_name=f"frame{idx:03d}.png"
                )
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
        save_path: str,
        device,
        progress: ProgressTracker,
        anchor2box,
        validation_dataloader,
    ):
        self.model = model
        self.device = device
        self.progress = progress
        self.save_path = save_path

        self.anchor2box = anchor2box
        self.nms = validation_cfg.nms
        self.validdataloader = validation_dataloader

    def solve(self):
        # logger.info("🧪 Start Validation!")
        self.model.eval()

        iou_thresholds = torch.arange(0.5, 1.0, 0.05)
        map_all = []
        self.progress.start_one_epoch(len(self.validdataloader))
        for data, targets in self.validdataloader:
            data, targets = data.to(self.device), targets.to(self.device)
            with torch.no_grad():
                raw_output = self.model(data)
            predict, _ = self.anchor2box(raw_output[0][3:], with_logits=True)

            nms_out = bbox_nms(predict, self.nms)
            for idx, predict in enumerate(nms_out):
                map_value = calculate_map(predict, targets[idx], iou_thresholds)
                map_all.append(map_value[0])
            self.progress.one_batch(mapp=torch.Tensor(map_all).mean())

        self.progress.finish_one_epoch()
