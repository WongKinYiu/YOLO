import torch
from torchvision.transforms import functional as TF


class Compose:
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes):
        for t in self.transforms:
            image, boxes = t(image, boxes)
        return image, boxes


class RandomHorizontalFlip:
    """Randomly horizontally flips the image along with the bounding boxes."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, boxes):
        if torch.rand(1) < self.p:
            image = TF.hflip(image)
            # Assuming boxes are in the format [cls, xmin, ymin, xmax, ymax]
            boxes[:, [1, 3]] = 1 - boxes[:, [3, 1]]
        return image, boxes
