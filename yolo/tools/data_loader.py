from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Generator, List, Tuple, Union

import numpy as np
import torch
from loguru import logger
from PIL import Image
from rich.progress import track
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from yolo.config.config import DataConfig, DatasetConfig
from yolo.tools.data_augmentation import *
from yolo.tools.data_augmentation import AugmentationComposer
from yolo.tools.dataset_preparation import prepare_dataset
from yolo.dataset.base import create_dataset, BaseDataset

def create_dataloader(data_cfg: DataConfig, dataset_cfg: DatasetConfig, task: str = "train", use_ddp: bool = False):
    if task == "inference":
        return StreamDataLoader(data_cfg)

    if dataset_cfg.auto_download:
        prepare_dataset(dataset_cfg, task)

    dataset = create_dataset(dataset_cfg, data_cfg, task)
    return YoloDataLoader(data_cfg, dataset, use_ddp)

class YoloDataLoader(DataLoader):
    def __init__(self, data_cfg: DataConfig, dataset: BaseDataset, use_ddp: bool = False):
        """Initializes the YoloDataLoader with hydra-config files."""
        sampler = DistributedSampler(dataset, shuffle=data_cfg.shuffle) if use_ddp else None
        self.image_size = data_cfg.image_size[0]
        super().__init__(
            dataset,
            batch_size=data_cfg.batch_size,
            sampler=sampler,
            shuffle=data_cfg.shuffle and not use_ddp,
            num_workers=data_cfg.cpu_num,
            pin_memory=data_cfg.pin_memory,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tensor]]:
        """
        A collate function to handle batching of images and their corresponding targets.

        Args:
            batch (list of tuples): Each tuple contains:
                - image (Tensor): The image tensor.
                - labels (Tensor): The tensor of labels for the image.

        Returns:
            Tuple[Tensor, List[Tensor]]: A tuple containing:
                - A tensor of batched images.
                - A list of tensors, each corresponding to bboxes for each image in the batch.
        """
        batch_size = len(batch)
        target_sizes = [item[1].size(0) for item in batch]
        # TODO: Improve readability of these proccess
        # TODO: remove maxBbox or reduce loss function memory usage
        batch_targets = torch.zeros(batch_size, min(max(target_sizes), 100), 5)
        batch_targets[:, :, 0] = -1
        for idx, target_size in enumerate(target_sizes):
            batch_targets[idx, : min(target_size, 100)] = batch[idx][1][:100]
        batch_targets[:, :, 1:] *= self.image_size

        batch_images, _, batch_reverse, batch_path = zip(*batch)
        batch_images = torch.stack(batch_images)
        batch_reverse = torch.stack(batch_reverse)

        return batch_size, batch_images, batch_targets, batch_reverse, batch_path

class StreamDataLoader:
    def __init__(self, data_cfg: DataConfig):
        self.source = data_cfg.source
        self.running = True
        self.is_stream = isinstance(self.source, int) or str(self.source).lower().startswith("rtmp://")

        self.transform = AugmentationComposer([], data_cfg.image_size)
        self.stop_event = Event()

        if self.is_stream:
            import cv2

            self.cap = cv2.VideoCapture(self.source)
        else:
            self.source = Path(self.source)
            self.queue = Queue()
            self.thread = Thread(target=self.load_source)
            self.thread.start()

    def load_source(self):
        if self.source.is_dir():  # image folder
            self.load_image_folder(self.source)
        elif any(self.source.suffix.lower().endswith(ext) for ext in [".mp4", ".avi", ".mkv"]):  # Video file
            self.load_video_file(self.source)
        else:  # Single image
            self.process_image(self.source)

    def load_image_folder(self, folder):
        folder_path = Path(folder)
        for file_path in folder_path.rglob("*"):
            if self.stop_event.is_set():
                break
            if file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                self.process_image(file_path)

    def process_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        if image is None:
            raise ValueError(f"Error loading image: {image_path}")
        self.process_frame(image)

    def load_video_file(self, video_path):
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            self.process_frame(frame)
        cap.release()

    def process_frame(self, frame):
        if isinstance(frame, np.ndarray):
            # TODO: we don't need cv2
            import cv2

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
        origin_frame = frame
        frame, _, rev_tensor = self.transform(frame, torch.zeros(0, 5))
        frame = frame[None]
        rev_tensor = rev_tensor[None]
        if not self.is_stream:
            self.queue.put((frame, rev_tensor, origin_frame))
        else:
            self.current_frame = (frame, rev_tensor, origin_frame)

    def __iter__(self) -> Generator[Tensor, None, None]:
        return self

    def __next__(self) -> Tensor:
        if self.is_stream:
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                raise StopIteration
            self.process_frame(frame)
            return self.current_frame
        else:
            try:
                frame = self.queue.get(timeout=1)
                return frame
            except Empty:
                raise StopIteration

    def stop(self):
        self.running = False
        if self.is_stream:
            self.cap.release()
        else:
            self.thread.join(timeout=1)

    def __len__(self):
        return self.queue.qsize() if not self.is_stream else 0
