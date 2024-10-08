import os
import time
from collections import defaultdict
from typing import Dict, Optional

import torch
from loguru import logger
from torch import Tensor, distributed
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from yolo.config.config import Config, DatasetConfig, TrainConfig, ValidationConfig
from yolo.model.yolo import YOLO
from yolo.tools.data_loader import StreamDataLoader, create_dataloader
from yolo.tools.drawer import draw_bboxes, draw_model
from yolo.tools.loss_functions import create_loss_function
from yolo.utils.bounding_box_utils import Vec2Box
from yolo.utils.logging_utils import ProgressLogger, log_model_structure
from yolo.utils.solver_utils import format_prediction, format_target
from yolo.utils.model_utils import (
    ExponentialMovingAverage,
    PostProccess,
    create_optimizer,
    create_scheduler,
)


class ModelTrainer:
    def __init__(self, cfg: Config, model: YOLO, vec2box: Vec2Box, progress: ProgressLogger, device, use_ddp: bool):
        train_cfg: TrainConfig = cfg.task
        self.model = model if not use_ddp else DDP(model, device_ids=[device])
        self.use_ddp = use_ddp
        self.vec2box = vec2box
        self.device = device
        self.optimizer = create_optimizer(model, train_cfg.optimizer)
        self.scheduler = create_scheduler(self.optimizer, train_cfg.scheduler)
        self.loss_fn = create_loss_function(cfg, vec2box)
        self.progress = progress
        self.num_epochs = cfg.task.epoch
        self.mAPs_dict = defaultdict(list)

        self.weights_dir = self.progress.save_path / "weights"
        self.weights_dir.mkdir(exist_ok=True)

        if not progress.quite_mode:
            log_model_structure(model.model)
            draw_model(model=model)

        self.validation_dataloader = create_dataloader(
            cfg.task.validation.data, cfg.dataset, cfg.task.validation.task, use_ddp
        )
        self.validator = ModelValidator(cfg.task.validation, model, vec2box, progress, device)

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
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss_item

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = defaultdict(float)
        total_samples = 0
        self.optimizer.next_epoch(len(dataloader))
        for batch_size, images, targets, _ in dataloader:
            self.optimizer.next_batch()
            loss_each = self.train_one_batch(images, targets)

            for loss_name, loss_val in loss_each.items():
                if self.use_ddp:  # collecting loss for each batch
                    distributed.all_reduce(loss_val, op=distributed.ReduceOp.AVG)
                total_loss[loss_name] += loss_val.item() * batch_size
            total_samples += batch_size
            self.progress.one_batch(loss_each)

        for loss_val in total_loss.values():
            loss_val /= total_samples

        if self.scheduler:
            self.scheduler.step()

        return total_loss

    def save_checkpoint(self, epoch_idx: int, file_name: Optional[str] = None):
        file_name = file_name or f"E{epoch_idx:03d}.pt"
        file_path = self.weights_dir / file_name

        checkpoint = {
            "epoch": epoch_idx,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.ema:
            self.ema.apply_shadow()
            checkpoint["model_state_dict_ema"] = self.model.state_dict()
            self.ema.restore()

        logger.info(f"💾 success save at {file_path}")
        torch.save(checkpoint, file_path)

    def good_epoch(self, mAPs: Dict[str, Tensor]) -> bool:
        save_flag = True
        for mAP_key, mAP_val in mAPs.items():
            self.mAPs_dict[mAP_key].append(mAP_val)
            if mAP_val < max(self.mAPs_dict[mAP_key]):
                save_flag = False
        return save_flag

    def solve(self, dataloader: DataLoader):
        logger.info("🚄 Start Training!")
        num_epochs = self.num_epochs

        self.progress.start_train(num_epochs)
        for epoch_idx in range(num_epochs):
            if self.use_ddp:
                dataloader.sampler.set_epoch(epoch_idx)

            self.progress.start_one_epoch(len(dataloader), "Train", self.optimizer, epoch_idx)
            epoch_loss = self.train_one_epoch(dataloader)
            self.progress.finish_one_epoch(epoch_loss, epoch_idx=epoch_idx)

            mAPs = self.validator.solve(self.validation_dataloader, epoch_idx=epoch_idx)
            if mAPs is not None and self.good_epoch(mAPs):
                self.save_checkpoint(epoch_idx=epoch_idx)
            # TODO: save model if result are better than before
        self.progress.finish_train()


class ModelTester:
    def __init__(self, cfg: Config, model: YOLO, vec2box: Vec2Box, progress: ProgressLogger, device):
        self.model = model
        self.device = device
        self.progress = progress

        self.post_proccess = PostProccess(vec2box, cfg.task.nms)
        self.save_path = progress.save_path / "images"
        os.makedirs(self.save_path, exist_ok=True)
        self.save_predict = getattr(cfg.task, "save_predict", None)
        self.idx2label = cfg.dataset.class_list

    def solve(self, dataloader: StreamDataLoader):
        logger.info("👀 Start Inference!")
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()

        if dataloader.is_stream:
            import cv2
            import numpy as np

            last_time = time.time()
        try:
            for idx, (images, rev_tensor, origin_frame) in enumerate(dataloader):
                images = images.to(self.device)
                rev_tensor = rev_tensor.to(self.device)
                with torch.no_grad():
                    predicts = self.model(images)
                    predicts = self.post_proccess(predicts, rev_tensor)
                img = draw_bboxes(origin_frame, predicts, idx2label=self.idx2label)

                if dataloader.is_stream:
                    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    fps = 1 / (time.time() - last_time)
                    cv2.putText(img, f"FPS: {fps:.2f}", (0, 15), 0, 0.5, (100, 255, 0), 1, cv2.LINE_AA)
                    last_time = time.time()
                    cv2.imshow("Prediction", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    if not self.save_predict:
                        continue
                if self.save_predict != False:
                    save_image_path = self.save_path / f"frame{idx:03d}.png"
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
        progress: ProgressLogger,
        device,
    ):
        self.model = model
        self.device = device
        self.progress = progress
        self.post_proccess = PostProccess(vec2box, validation_cfg.nms)

    def solve(self, dataloader, epoch_idx=1):
        logger.info("🧪 Start Validation!")
        metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
        self.model.eval()
        self.progress.start_one_epoch(len(dataloader), task="Validate")
        for _, images, targets, rev_tensor in dataloader:
            images, targets, rev_tensor = images.to(self.device), targets.to(self.device), rev_tensor.to(self.device)
            with torch.no_grad():
                predicts = self.model(images)
                predicts = self.post_proccess(predicts)
                batch_metrics = metric([format_prediction(predict) for predict in predicts], 
                       [format_target(target) for target in targets])
            
            self.progress.one_batch({
                "map": batch_metrics["map"],
                "map_50": batch_metrics["map_50"],
            })

        epoch_metrics = metric.compute()
        del epoch_metrics['classes']
        self.progress.finish_one_epoch(epoch_metrics, epoch_idx=epoch_idx)
        self.progress.visualize_image(images, targets, predicts, epoch_idx=epoch_idx)
        return epoch_metrics
