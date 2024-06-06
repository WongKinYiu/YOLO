import os
from os import path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Generator, List, Tuple, Union

import cv2
import hydra
import numpy as np
import torch
from loguru import logger
from PIL import Image
from rich.progress import track
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from yolo.config.config import DataConfig, DatasetConfig
from yolo.tools.data_augmentation import (
    AugmentationComposer,
    HorizontalFlip,
    MixUp,
    Mosaic,
    VerticalFlip,
)
from yolo.tools.dataset_preparation import prepare_dataset
from yolo.tools.drawer import draw_bboxes
from yolo.utils.dataset_utils import (
    create_image_metadata,
    locate_label_paths,
    scale_segmentation,
)


class YoloDataset(Dataset):
    def __init__(self, data_cfg: DataConfig, dataset_cfg: DatasetConfig, phase: str = "train2017"):
        augment_cfg = data_cfg.data_augment
        self.image_size = data_cfg.image_size
        phase_name = dataset_cfg.get(phase, phase)

        transforms = [eval(aug)(prob) for aug, prob in augment_cfg.items()]
        self.transform = AugmentationComposer(transforms, self.image_size)
        self.transform.get_more_data = self.get_more_data
        self.data = self.load_data(dataset_cfg.path, phase_name)

    def load_data(self, dataset_path, phase_name):
        """
        Loads data from a cache or generates a new cache for a specific dataset phase.

        Parameters:
            dataset_path (str): The root path to the dataset directory.
            phase_name (str): The specific phase of the dataset (e.g., 'train', 'test') to load or generate data for.

        Returns:
            dict: The loaded data from the cache for the specified phase.
        """
        cache_path = path.join(dataset_path, f"{phase_name}.cache")

        if not path.isfile(cache_path):
            logger.info("🏭 Generating {} cache", phase_name)
            data = self.filter_data(dataset_path, phase_name)
            torch.save(data, cache_path)
        else:
            data = torch.load(cache_path)
            logger.info("📦 Loaded {} cache", phase_name)
        return data

    def filter_data(self, dataset_path: str, phase_name: str) -> list:
        """
        Filters and collects dataset information by pairing images with their corresponding labels.

        Parameters:
            images_path (str): Path to the directory containing image files.
            labels_path (str): Path to the directory containing label files.

        Returns:
            list: A list of tuples, each containing the path to an image file and its associated segmentation as a tensor.
        """
        images_path = path.join(dataset_path, "images", phase_name)
        labels_path, data_type = locate_label_paths(dataset_path, phase_name)
        images_list = sorted(os.listdir(images_path))
        if data_type == "json":
            annotations_index, image_info_dict = create_image_metadata(labels_path)

        data = []
        valid_inputs = 0
        for image_name in track(images_list, description="Filtering data"):
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            image_id, _ = path.splitext(image_name)

            if data_type == "json":
                image_info = image_info_dict.get(image_id, None)
                if image_info is None:
                    continue
                annotations = annotations_index.get(image_info["id"], [])
                image_seg_annotations = scale_segmentation(annotations, image_info)
                if not image_seg_annotations:
                    continue

            elif data_type == "txt":
                label_path = path.join(labels_path, f"{image_id}.txt")
                if not path.isfile(label_path):
                    continue
                with open(label_path, "r") as file:
                    image_seg_annotations = [list(map(float, line.strip().split())) for line in file]
            else:
                image_seg_annotations = []

            labels = self.load_valid_labels(image_id, image_seg_annotations)

            img_path = path.join(images_path, image_name)
            data.append((img_path, labels))
            valid_inputs += 1
        logger.info("Recorded {}/{} valid inputs", valid_inputs, len(images_list))
        return data

    def load_valid_labels(self, label_path, seg_data_one_img) -> Union[torch.Tensor, None]:
        """
        Loads and validates bounding box data is [0, 1] from a label file.

        Parameters:
            label_path (str): The filepath to the label file containing bounding box data.

        Returns:
            torch.Tensor or None: A tensor of all valid bounding boxes if any are found; otherwise, None.
        """
        bboxes = []
        for seg_data in seg_data_one_img:
            cls = seg_data[0]
            points = np.array(seg_data[1:]).reshape(-1, 2)
            valid_points = points[(points >= 0) & (points <= 1)].reshape(-1, 2)
            if valid_points.size > 1:
                bbox = torch.tensor([cls, *valid_points.min(axis=0), *valid_points.max(axis=0)])
                bboxes.append(bbox)

        if bboxes:
            return torch.stack(bboxes)
        else:
            logger.warning("No valid BBox in {}", label_path)
            return torch.zeros((0, 5))

    def get_data(self, idx):
        img_path, bboxes = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        return img, bboxes

    def get_more_data(self, num: int = 1):
        indices = torch.randint(0, len(self), (num,))
        return [self.get_data(idx) for idx in indices]

    def __getitem__(self, idx) -> Union[Image.Image, torch.Tensor]:
        img, bboxes = self.get_data(idx)
        img, bboxes = self.transform(img, bboxes)
        return img, bboxes

    def __len__(self) -> int:
        return len(self.data)


class YoloDataLoader(DataLoader):
    def __init__(self, data_cfg: DataConfig, dataset_cfg: DatasetConfig, task: str = "train"):
        """Initializes the YoloDataLoader with hydra-config files."""
        dataset = YoloDataset(data_cfg, dataset_cfg, task)
        self.image_size = data_cfg.image_size[0]
        super().__init__(
            dataset,
            batch_size=data_cfg.batch_size,
            shuffle=data_cfg.shuffle,
            num_workers=data_cfg.cpu_num,
            pin_memory=data_cfg.pin_memory,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        A collate function to handle batching of images and their corresponding targets.

        Args:
            batch (list of tuples): Each tuple contains:
                - image (torch.Tensor): The image tensor.
                - labels (torch.Tensor): The tensor of labels for the image.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: A tuple containing:
                - A tensor of batched images.
                - A list of tensors, each corresponding to bboxes for each image in the batch.
        """
        batch_size = len(batch)
        target_sizes = [item[1].size(0) for item in batch]
        # TODO: Improve readability of these proccess
        batch_targets = torch.zeros(batch_size, max(target_sizes), 5)
        batch_targets[:, :, 0] = -1
        for idx, target_size in enumerate(target_sizes):
            batch_targets[idx, :target_size] = batch[idx][1]
        batch_targets[:, :, 1:] *= self.image_size

        batch_images = torch.stack([item[0] for item in batch])

        return batch_images, batch_targets


def create_dataloader(data_cfg: DataConfig, dataset_cfg: DatasetConfig, task: str = "train"):
    if task == "inference":
        return StreamDataLoader(data_cfg)

    if dataset_cfg.auto_download:
        prepare_dataset(dataset_cfg)

    return YoloDataLoader(data_cfg, dataset_cfg, task)


class StreamDataLoader:
    def __init__(self, data_cfg: DataConfig):
        self.source = data_cfg.source
        self.running = True
        self.is_stream = isinstance(self.source, int) or self.source.lower().startswith("rtmp://")

        self.transform = AugmentationComposer([], data_cfg.image_size)
        self.stop_event = Event()

        if self.is_stream:
            self.cap = cv2.VideoCapture(self.source)
        else:
            self.queue = Queue()
            self.thread = Thread(target=self.load_source)
            self.thread.start()

    def load_source(self):
        if os.path.isdir(self.source):  # image folder
            self.load_image_folder(self.source)
        elif any(self.source.lower().endswith(ext) for ext in [".mp4", ".avi", ".mkv"]):  # Video file
            self.load_video_file(self.source)
        else:  # Single image
            self.process_image(self.source)

    def load_image_folder(self, folder):
        for root, _, files in os.walk(folder):
            for file in files:
                if self.stop_event.is_set():
                    break
                if any(file.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp"]):
                    self.process_image(os.path.join(root, file))

    def process_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        if image is None:
            raise ValueError(f"Error loading image: {image_path}")
        self.process_frame(image)

    def load_video_file(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            self.process_frame(frame)
        cap.release()

    def cv2_to_tensor(self, frame: np.ndarray) -> Tensor:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_float = frame_rgb.astype("float32") / 255.0
        return torch.from_numpy(frame_float).permute(2, 0, 1)[None]

    def process_frame(self, frame):
        if isinstance(frame, np.ndarray):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
        frame, _ = self.transform(frame, torch.zeros(0, 5))
        frame = frame[None]
        if not self.is_stream:
            self.queue.put(frame)
        else:
            self.current_frame = frame

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
