import json
import os
from itertools import chain
from os import listdir, path
from typing import Any, Dict, List, Optional, Tuple, Union

import diskcache as dc
import hydra
import numpy as np
import torch
from data_augment import Compose, HorizontalFlip, MixUp, Mosaic, VerticalFlip
from drawer import draw_bboxes
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm.rich import tqdm


def find_labels_path(dataset_path: str, phase_name: str):
    """
    Find the path to label files for a specified dataset and phase(e.g. training).

    Args:
        dataset_path (str): The path to the root directory of the dataset.
        phase_name (str): The name of the phase for which labels are being searched (e.g., "train", "val", "test").

    Returns:
        Tuple[str, str]: A tuple containing the path to the labels file and the file format ("json" or "txt").
    """
    json_labels_path = path.join(dataset_path, "annotations", f"instances_{phase_name}.json")

    txt_labels_path = path.join(dataset_path, "label", phase_name)

    if path.isfile(json_labels_path):
        return json_labels_path, "json"

    elif path.isdir(txt_labels_path):
        txt_files = [f for f in os.listdir(txt_labels_path) if f.endswith(".txt")]
        if txt_files:
            return txt_labels_path, "txt"

    raise FileNotFoundError("No labels found in the specified dataset path and phase name.")


def create_image_info_dict(labels_path: str) -> Tuple[Dict[str, List], Dict[str, Dict]]:
    """
    Create a dictionary containing image information and annotations indexed by image ID.

    Args:
        labels_path (str): The path to the annotation json file.

    Returns:
        - annotations_index: A dictionary where keys are image IDs and values are lists of annotations.
        - image_info_dict: A dictionary where keys are image file names without extension and values are image information dictionaries.
    """
    with open(labels_path, "r") as file:
        labels_data = json.load(file)
        annotations_index = index_annotations_by_image(labels_data)  # check lookup is a good name?
        image_info_dict = {path.splitext(img["file_name"])[0]: img for img in labels_data["images"]}
        return annotations_index, image_info_dict


def index_annotations_by_image(data: Dict[str, Any]):
    """
    Use image index to lookup every annotations
    Args:
        data (Dict[str, Any]): A dictionary containing annotation data.

    Returns:
        Dict[int, List[Dict[str, Any]]]: A dictionary where keys are image IDs and values are lists of annotations.
        Annotations with "iscrowd" set to True are excluded from the index.

    """
    annotation_lookup = {}
    for anno in data["annotations"]:
        if anno["iscrowd"]:
            continue
        image_id = anno["image_id"]
        if image_id not in annotation_lookup:
            annotation_lookup[image_id] = []
        annotation_lookup[image_id].append(anno)
    return annotation_lookup


def get_scaled_segmentation(
    annotations: List[Dict[str, Any]], image_dimensions: Dict[str, int]
) -> Optional[List[List[float]]]:
    """
    Scale the segmentation data based on image dimensions and return a list of scaled segmentation data.

    Args:
        annotations (List[Dict[str, Any]]): A list of annotation dictionaries.
        image_dimensions (Dict[str, int]): A dictionary containing image dimensions (height and width).

    Returns:
        Optional[List[List[float]]]: A list of scaled segmentation data, where each sublist contains category_id followed by scaled (x, y) coordinates.
    """
    if annotations is None:
        return None

    seg_array_with_cat = []
    h, w = image_dimensions["height"], image_dimensions["width"]
    for anno in annotations:
        category_id = anno["category_id"]
        seg_list = [item for sublist in anno["segmentation"] for item in sublist]
        scaled_seg_data = (
            np.array(seg_list).reshape(-1, 2) / [w, h]
        ).tolist()  # make the list group in x, y pairs and scaled with image width, height
        scaled_flat_seg_data = [category_id] + list(chain(*scaled_seg_data))  # flatten the scaled_seg_data list
        seg_array_with_cat.append(scaled_flat_seg_data)

    return seg_array_with_cat


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
            data = self.filter_data(dataset_path, phase_name)
            cache[phase_name] = data

        cache.close()
        logger.info("📦 Loaded {} cache", phase_name)
        data = cache[phase_name]
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
        labels_path, data_type = find_labels_path(dataset_path, phase_name)
        images_list = sorted(os.listdir(images_path))
        if data_type == "json":
            annotations_index, image_info_dict = create_image_info_dict(labels_path)

        data = []
        valid_inputs = 0
        for image_name in tqdm(images_list, desc="Filtering data"):
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            image_id, _ = path.splitext(image_name)

            if data_type == "json":
                image_info = image_info_dict.get(image_id, None)
                if image_info is None:
                    continue
                annotations = annotations_index.get(image_info["id"], [])
                image_seg_annotations = get_scaled_segmentation(annotations, image_info)
                if not image_seg_annotations:
                    continue

            elif data_type == "txt":
                label_path = path.join(labels_path, f"{image_id}.txt")
                if not path.isfile(label_path):
                    continue
                with open(label_path, "r") as file:
                    image_seg_annotations = [
                        list(map(float, line.strip().split())) for line in file
                    ]  # add a comment for this line, complicated, do you need "list", im not sure

            labels = self.load_valid_labels(image_id, image_seg_annotations)
            if labels is not None:
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
