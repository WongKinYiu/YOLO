from os import listdir, path
from typing import List, Tuple, Union

import diskcache as dc
import hydra
import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm.rich import tqdm

from utils.data_augment import Compose, HorizontalFlip, MixUp, Mosaic, VerticalFlip
from utils.drawer import draw_bboxes


class YoloDataset(Dataset):
    def __init__(self, config: dict, phase: str = "train2017", image_size: int = 640):
        dataset_cfg = config.data
        augment_cfg = config.augmentation
        phase_name = dataset_cfg.get(phase, phase)
        self.image_size = image_size

        transforms = [eval(aug)(prob) for aug, prob in augment_cfg.items()]
        self.transform = Compose(transforms, self.image_size)
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
        cache_path = path.join(dataset_path, ".cache")
        cache = dc.Cache(cache_path)
        data = cache.get(phase_name)

        if data is None:
            logger.info("Generating {} cache", phase_name)
            images_path = path.join(dataset_path, "images", phase_name)
            labels_path = path.join(dataset_path, "labels", phase_name)
            data = self.filter_data(images_path, labels_path)
            cache[phase_name] = data

        cache.close()
        logger.info("📦 Loaded {} cache", phase_name)
        data = cache[phase_name]
        return data

    def filter_data(self, images_path: str, labels_path: str) -> list:
        """
        Filters and collects dataset information by pairing images with their corresponding labels.

        Parameters:
            images_path (str): Path to the directory containing image files.
            labels_path (str): Path to the directory containing label files.

        Returns:
            list: A list of tuples, each containing the path to an image file and its associated labels as a tensor.
        """
        data = []
        valid_inputs = 0
        images_list = sorted(listdir(images_path))
        for image_name in tqdm(images_list, desc="Filtering data"):
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = path.join(images_path, image_name)
            base_name, _ = path.splitext(image_name)
            label_path = path.join(labels_path, f"{base_name}.txt")

            if path.isfile(label_path):
                labels = self.load_valid_labels(label_path)
                if labels is not None:
                    data.append((img_path, labels))
                    valid_inputs += 1

        logger.info("Recorded {}/{} valid inputs", valid_inputs, len(images_list))
        return data

    def load_valid_labels(self, label_path: str) -> Union[torch.Tensor, None]:
        """
        Loads and validates bounding box data is [0, 1] from a label file.

        Parameters:
            label_path (str): The filepath to the label file containing bounding box data.

        Returns:
            torch.Tensor or None: A tensor of all valid bounding boxes if any are found; otherwise, None.
        """
        bboxes = []
        with open(label_path, "r") as file:
            for line in file:
                parts = list(map(float, line.strip().split()))
                cls = parts[0]
                points = np.array(parts[1:]).reshape(-1, 2)
                valid_points = points[(points >= 0) & (points <= 1)].reshape(-1, 2)
                if valid_points.size > 1:
                    bbox = torch.tensor([cls, *valid_points.min(axis=0), *valid_points.max(axis=0)])
                    bboxes.append(bbox)

        if bboxes:
            return torch.stack(bboxes)
        else:
            logger.warning("No valid BBox in {}", label_path)
            return None

    def get_data(self, idx):
        img_path, bboxes = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        return img, bboxes

    def get_more_data(self, num: int = 1):
        indices = torch.randint(0, len(self), (num,))
        return [self.get_data(idx) for idx in indices]

    def __getitem__(self, idx) -> Union[Image.Image, torch.Tensor]:
        img, bboxes = self.get_data(idx)
        if self.transform:
            img, bboxes = self.transform(img, bboxes)
        img = TF.to_tensor(img)
        return img, bboxes

    def __len__(self) -> int:
        return len(self.data)


class YoloDataLoader(DataLoader):
    def __init__(self, config: dict):
        """Initializes the YoloDataLoader with hydra-config files."""
        hyper = config.hyper.data
        dataset = YoloDataset(config)

        super().__init__(
            dataset,
            batch_size=hyper.batch_size,
            shuffle=hyper.shuffle,
            num_workers=hyper.num_workers,
            pin_memory=hyper.pin_memory,
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
        images = torch.stack([item[0] for item in batch])
        targets = [item[1] for item in batch]
        return images, targets


def get_dataloader(config):
    return YoloDataLoader(config)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    dataloader = get_dataloader(cfg)
    draw_bboxes(*next(iter(dataloader)))


if __name__ == "__main__":
    import sys

    sys.path.append("./")
    from tools.log_helper import custom_logger

    custom_logger()
    main()
