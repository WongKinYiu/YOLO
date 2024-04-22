from PIL import Image
from os import path
import os
import hydra
import numpy as np
import torch
from torch.utils.data import Dataset
from loguru import logger
from tqdm.rich import tqdm
import diskcache as dc


class YoloDataset(Dataset):
    def __init__(self, dataset_cfg: dict, phase="train", transform=None, mixup=None):
        phase_name = dataset_cfg.get(phase, phase)

        self.transform = transform
        self.mixup = mixup
        self.data = self.load_data(dataset_cfg.path, phase_name)

    def load_data(self, dataset_path, phase_name):
        cache = dc.Cache(path.join(dataset_path, ".cache"))

        if phase_name not in cache:
            logger.info("Generate {} Cache", phase_name)

            images_path = path.join(dataset_path, phase_name, "images")
            labels_path = path.join(dataset_path, phase_name, "labels")

            cache[phase_name] = self.filter_data(images_path, labels_path)

        logger.info("Load {} Cache", phase_name)
        data = cache[phase_name]
        cache.close()

        return data

    def filter_data(self, images_path, labels_path):
        data = []
        valid_input = 0
        images_list = os.listdir(images_path)
        images_list.sort()
        for image_name in tqdm(images_list):
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = path.join(images_path, image_name)
            base_name, _ = path.splitext(image_name)
            label_name = base_name + ".txt"
            label_path = path.join(labels_path, label_name)

            if not path.isfile(label_path):
                # logger.warning(f"Warning: No label file for {label_path}")
                continue

            labels = self.load_valid_labels(label_path)
            if labels is not None:
                data.append((img_path, labels))
                valid_input += 1
        logger.info("Finish Record {}/{}", valid_input, len(os.listdir(images_path)))
        return data

    def load_valid_labels(self, label_path):
        bboxes = []
        with open(label_path, "r") as file:
            for line in file:
                segment = list(map(float, line.strip().split()))
                cls = segment[0]
                # Ensure parts length is odd and more than two points
                if len(segment) % 2 != 1 or len(segment) < 5:
                    logger.warning(f"Warning: Format error in {label_path}")
                    continue
                points = np.array(segment[1:]).reshape(-1, 2)  # change points to n x 2
                valid_idx = np.any((points <= 1) | (points >= 0), axis=1)  # filter outlier points
                points = points[valid_idx]  # only keep valid points

                bbox = torch.tensor([cls, *points.max(axis=0), *points.min(axis=0)])
                bboxes.append(bbox)
        if not bboxes:
            logger.warning(f"Warning: No valid BBox in {label_path}")
            return None
        return torch.stack(bboxes)

    def __getitem__(self, idx):
        img_path, bboxes = self.data[idx]
        img = Image.open(img_path).convert("RGB")

        return img, bboxes

    def __len__(self):
        return len(self.images)


@hydra.main(config_path="../config/data", config_name="coco", version_base=None)
def main(cfg):
    dataset = YoloDataset(cfg)


if __name__ == "__main__":
    import sys

    sys.path.append("./")
    from tools.log_helper import custom_logger

    custom_logger()
    main()
