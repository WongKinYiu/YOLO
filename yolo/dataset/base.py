from __future__ import annotations
from typing import List

import hydra
from torch.utils.data import Dataset
import torch

from yolo.config.config import DataConfig, DatasetConfig
from yolo.tools.data_augmentation import AugmentationComposer
from yolo.tools.dataset_preparation import prepare_dataset
from yolo.utils import primitives



def create_dataset(dataset_cfg: DatasetConfig, data_cfg: DataConfig, task: str) -> BaseDataset:
    if task == "train":
        init_config = dataset_cfg.train_dataset
    elif task == "validation":
        init_config = dataset_cfg.validation_dataset
    else:
        raise ValueError(f"Invalid task: {task}")
    return hydra.utils.instantiate(init_config, data_cfg=data_cfg)
    

class BaseDataset(Dataset):
    def __init__(self, data_cfg: DataConfig):
        self.image_size = data_cfg.image_size        
        self.transform = AugmentationComposer(data_cfg.data_augment, self.image_size)
        self.transform.get_more_data = self.get_more_data
        self.load_data()

    def extract_data(self) -> List[primitives.YoloImage]:
        """ Prepare the data for the dataset.
        Args:
            dataset_cfg: The dataset configuration.
            
        Returns:
            List[Image]: The list of images.
        """
        raise NotImplementedError
    
    def load_data(self) -> None:
        self.data = self.extract_data()
    
    def extract_sample(self, idx: int) -> tuple[primitives.YoloImage, list[torch.Tensor]]:
        sample = self.data[idx]
        image = sample.image
        image_path = sample.image_path
        
        bboxes = torch.stack([bbox.tensor for bbox in sample.bboxes])
        return image, bboxes, image_path
    
    def __getitem__(self, idx: int) -> primitives.YoloImage:
        image, bboxes, image_path = self.extract_sample(idx)
        image, bboxes, rev_tensor = self.transform(image, bboxes)
        return image, bboxes, rev_tensor, image_path

    def __len__(self) -> int:
        return len(self.data)
    
    def get_more_data(self, num: int = 1) -> List[primitives.YoloImage]:
        indices = torch.randint(0, len(self), (num,))
        return [self.extract_sample(idx)[:2] for idx in indices]
    