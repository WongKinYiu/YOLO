from pathlib import Path
from queue import Empty, Queue
from statistics import mean
from threading import Event, Thread
from typing import Generator, List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from rich.progress import track
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from yolo.config.config import DataConfig, DatasetConfig
from yolo.tools.data_augmentation import *
from yolo.tools.data_augmentation import AugmentationComposer
from yolo.tools.dataset_preparation import prepare_dataset
from yolo.utils.bounding_box_utils import calculate_iou
from yolo.utils.dataset_utils import (
    create_image_metadata,
    locate_label_paths,
    scale_segmentation,
    tensorlize,
)
from yolo.utils.logger import logger


class YoloDataset(Dataset):
    def __init__(self, data_cfg: DataConfig, dataset_cfg: DatasetConfig, phase: str = "train2017"):
        augment_cfg = data_cfg.data_augment
        self.image_size = data_cfg.image_size
        phase_name = dataset_cfg.get(phase, phase)
        self.batch_size = data_cfg.batch_size
        self.dynamic_shape = getattr(data_cfg, "dynamic_shape", False)
        self.base_size = mean(self.image_size)

        transforms = [eval(aug)(prob) for aug, prob in augment_cfg.items()]
        self.transform = AugmentationComposer(transforms, self.image_size, self.base_size)
        self.transform.get_more_data = self.get_more_data
        self.img_paths, self.bboxes, self.ratios = tensorlize(self.load_data(Path(dataset_cfg.path), phase_name))

    def load_data(self, dataset_path: Path, phase_name: str):
        """
        Loads data from a cache or generates a new cache for a specific dataset phase.

        Parameters:
            dataset_path (Path): The root path to the dataset directory.
            phase_name (str): The specific phase of the dataset (e.g., 'train', 'test') to load or generate data for.

        Returns:
            dict: The loaded data from the cache for the specified phase.
        """
        cache_path = dataset_path / f"{phase_name}.cache"

        if not cache_path.exists():
            logger.info(f":factory: Generating {phase_name} cache")
            data = self.filter_data(dataset_path, phase_name, self.dynamic_shape)
            torch.save(data, cache_path)
        else:
            try:
                data = torch.load(cache_path, weights_only=False)
            except Exception as e:
                logger.error(
                    f":rotating_light: Failed to load the cache at '{cache_path}'.\n"
                    ":rotating_light: This may be caused by using cache from different other YOLO.\n"
                    ":rotating_light: Please clean the cache and try running again."
                )
                raise e
            logger.info(f":package: Loaded {phase_name} cache")
        return data

    def filter_data(self, dataset_path: Path, phase_name: str, sort_image: bool = False) -> list:
        """
        Filters and collects dataset information by pairing images with their corresponding labels.

        Parameters:
            images_path (Path): Path to the directory containing image files.
            labels_path (str): Path to the directory containing label files.
            sort_image (bool): If True, sorts the dataset by the width-to-height ratio of images in descending order.

        Returns:
            list: A list of tuples, each containing the path to an image file and its associated segmentation as a tensor.
        """
        images_path = dataset_path / "images" / phase_name
        labels_path, data_type = locate_label_paths(dataset_path, phase_name)
        images_list = sorted([p.name for p in Path(images_path).iterdir() if p.is_file()])
        if data_type == "json":
            annotations_index, image_info_dict = create_image_metadata(labels_path)

        data = []
        valid_inputs = 0
        for image_name in track(images_list, description="Filtering data"):
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            image_id = Path(image_name).stem

            if data_type == "json":
                image_info = image_info_dict.get(image_id, None)
                if image_info is None:
                    continue
                annotations = annotations_index.get(image_info["id"], [])
                image_seg_annotations = scale_segmentation(annotations, image_info)
            elif data_type == "txt":
                label_path = labels_path / f"{image_id}.txt"
                if not label_path.is_file():
                    continue
                with open(label_path, "r") as file:
                    image_seg_annotations = [list(map(float, line.strip().split())) for line in file]
            else:
                image_seg_annotations = []

            labels = self.load_valid_labels(image_id, image_seg_annotations)

            if labels is not None and len(labels) > 1:
                labels = self.deduplicate_labels(labels)

            img_path = images_path / image_name
            if sort_image:
                with Image.open(img_path) as img:
                    width, height = img.size
            else:
                width, height = 0, 1
            data.append((img_path, labels, width / height))
            valid_inputs += 1

        data = sorted(data, key=lambda x: x[2], reverse=True)

        logger.info(f"Recorded {valid_inputs}/{len(images_list)} valid inputs")
        return data

    def deduplicate_labels(self, labels: Tensor) -> Tensor:
        """
        Removes duplicate labels from a Tensor of bboxes.

        Parameters:
            labels (Tensor): A tensor of all input bounding boxes.

        Returns:
            Tensor: A tensor of all remaining bounding boxes.
        """
        dedup_labels = []

        for l in labels:
            acceptable = True
            for ddl in dedup_labels:
                if int(l[0]) != int(ddl[0]):
                    continue

                if float(calculate_iou(l[1:], ddl[1:])) > .99:
                    acceptable = False
                    break

            if acceptable:
                dedup_labels.append(l)

        return torch.stack(dedup_labels)

    def load_valid_labels(self, label_path: str, seg_data_one_img: list) -> Union[Tensor, None]:
        """
        Loads and validates bounding box data is [0, 1] from a label file.

        Parameters:
            label_path (str): The filepath to the label file containing bounding box data.

        Returns:
            Tensor or None: A tensor of all valid bounding boxes if any are found; otherwise, None.
        """
        bboxes = []
        try:
            for seg_data in seg_data_one_img:
                cls = seg_data[0]
                points = np.array(seg_data[1:]).reshape(-1, 2)
                valid_points = points[(points >= 0) & (points <= 1)].reshape(-1, 2)
                if valid_points.size > 1:
                    bbox = torch.tensor([cls, *valid_points.min(axis=0), *valid_points.max(axis=0)])
                    bboxes.append(bbox)
        except ValueError:
            logger.warning("Invalid BBox in {}", label_path)
            return torch.zeros((0, 5))

        if bboxes:
            return torch.stack(bboxes)
        else:
            logger.warning(f"No valid BBox in {label_path}")
            return torch.zeros((0, 5))

    def get_data(self, idx):
        img_path, bboxes = self.img_paths[idx], self.bboxes[idx]
        valid_mask = bboxes[:, 0] != -1
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        return img, torch.from_numpy(bboxes[valid_mask]), img_path

    def get_more_data(self, num: int = 1):
        indices = torch.randint(0, len(self), (num,))
        return [self.get_data(idx)[:2] for idx in indices]

    def _update_image_size(self, idx: int) -> None:
        """Update image size based on dynamic shape and batch settings."""
        batch_start_idx = (idx // self.batch_size) * self.batch_size
        image_ratio = self.ratios[batch_start_idx].clip(1 / 3, 3)
        shift = ((self.base_size / 32 * (image_ratio - 1)) // (image_ratio + 1)) * 32

        self.image_size = [int(self.base_size + shift), int(self.base_size - shift)]
        self.transform.pad_resize.set_size(self.image_size)

    def __getitem__(self, idx) -> Tuple[Image.Image, Tensor, Tensor, List[str]]:
        img, bboxes, img_path = self.get_data(idx)

        if self.dynamic_shape:
            self._update_image_size(idx)

        img, bboxes, rev_tensor = self.transform(img, bboxes)
        bboxes[:, [1, 3]] *= self.image_size[0]
        bboxes[:, [2, 4]] *= self.image_size[1]
        return img, bboxes, rev_tensor, img_path

    def __len__(self) -> int:
        return len(self.bboxes)


def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tensor]]:
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
    # TODO: Improve readability of these process
    # TODO: remove maxBbox or reduce loss function memory usage
    batch_targets = torch.zeros(batch_size, min(max(target_sizes), 100), 5)
    batch_targets[:, :, 0] = -1
    for idx, target_size in enumerate(target_sizes):
        batch_targets[idx, : min(target_size, 100)] = batch[idx][1][:100]

    batch_images, _, batch_reverse, batch_path = zip(*batch)
    batch_images = torch.stack(batch_images)
    batch_reverse = torch.stack(batch_reverse)

    return batch_size, batch_images, batch_targets, batch_reverse, batch_path


def create_dataloader(data_cfg: DataConfig, dataset_cfg: DatasetConfig, task: str = "train"):
    if task == "inference":
        return StreamDataLoader(data_cfg)

    if getattr(dataset_cfg, "auto_download", False):
        prepare_dataset(dataset_cfg, task)
    dataset = YoloDataset(data_cfg, dataset_cfg, task)

    return DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.cpu_num,
        pin_memory=data_cfg.pin_memory,
        collate_fn=collate_fn,
    )


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
