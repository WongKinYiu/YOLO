import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


class AugmentationComposer:
    """Composes several transforms together."""

    def __init__(self, transforms, image_size: int = [640, 640]):
        self.transforms = transforms
        # TODO: handle List of image_size [640, 640]
        self.image_size = image_size[0]
        self.pad_resize = PadAndResize(self.image_size)

        for transform in self.transforms:
            if hasattr(transform, "set_parent"):
                transform.set_parent(self)

    def __call__(self, image, boxes=torch.zeros(0, 5)):
        for transform in self.transforms:
            image, boxes = transform(image, boxes)
        image, boxes, rev_tensor = self.pad_resize(image, boxes)
        image = TF.to_tensor(image)
        return image, boxes, rev_tensor


# TODO: RandomCrop, Resize, ... etc.


class PadAndResize:
    def __init__(self, image_size):
        """Initialize the object with the target image size."""
        self.image_size = image_size

    def __call__(self, image, boxes):
        original_size = max(image.size)
        scale = self.image_size / original_size
        square_img = Image.new("RGB", (original_size, original_size), (128, 128, 128))
        left = (original_size - image.width) // 2
        top = (original_size - image.height) // 2
        square_img.paste(image, (left, top))

        resized_img = square_img.resize((self.image_size, self.image_size))

        boxes[:, 1] = (boxes[:, 1] * image.width + left) / self.image_size * scale
        boxes[:, 2] = (boxes[:, 2] * image.height + top) / self.image_size * scale
        boxes[:, 3] = (boxes[:, 3] * image.width + left) / self.image_size * scale
        boxes[:, 4] = (boxes[:, 4] * image.height + top) / self.image_size * scale

        rev_tensor = torch.tensor([scale, left, top, left, top])
        return resized_img, boxes, rev_tensor


class HorizontalFlip:
    """Randomly horizontally flips the image along with the bounding boxes."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            image = TF.hflip(image)
            boxes[:, [1, 3]] = 1 - boxes[:, [3, 1]]
        return image, boxes


class VerticalFlip:
    """Randomly vertically flips the image along with the bounding boxes."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            image = TF.vflip(image)
            boxes[:, [2, 4]] = 1 - boxes[:, [4, 2]]
        return image, boxes


class Mosaic:
    """Applies the Mosaic augmentation to a batch of images and their corresponding boxes."""

    def __init__(self, prob=0.5):
        self.prob = prob
        self.parent = None

    def set_parent(self, parent):
        self.parent = parent

    def __call__(self, image, boxes):
        if torch.rand(1) >= self.prob:
            return image, boxes

        assert self.parent is not None, "Parent is not set. Mosaic cannot retrieve image size."

        img_sz = self.parent.image_size  # Assuming `image_size` is defined in parent
        more_data = self.parent.get_more_data(3)  # get 3 more images randomly

        data = [(image, boxes)] + more_data
        mosaic_image = Image.new("RGB", (2 * img_sz, 2 * img_sz))
        vectors = np.array([(-1, -1), (0, -1), (-1, 0), (0, 0)])
        center = np.array([img_sz, img_sz])
        all_labels = []

        for (image, boxes), vector in zip(data, vectors):
            this_w, this_h = image.size
            coord = tuple(center + vector * np.array([this_w, this_h]))

            mosaic_image.paste(image, coord)
            xmin, ymin, xmax, ymax = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
            xmin = (xmin * this_w + coord[0]) / (2 * img_sz)
            xmax = (xmax * this_w + coord[0]) / (2 * img_sz)
            ymin = (ymin * this_h + coord[1]) / (2 * img_sz)
            ymax = (ymax * this_h + coord[1]) / (2 * img_sz)

            adjusted_boxes = torch.stack([boxes[:, 0], xmin, ymin, xmax, ymax], dim=1)
            all_labels.append(adjusted_boxes)

        all_labels = torch.cat(all_labels, dim=0)
        mosaic_image = mosaic_image.resize((img_sz, img_sz))
        return mosaic_image, all_labels


class MixUp:
    """Applies the MixUp augmentation to a pair of images and their corresponding boxes."""

    def __init__(self, prob=0.5, alpha=1.0):
        self.alpha = alpha
        self.prob = prob
        self.parent = None

    def set_parent(self, parent):
        """Set the parent dataset object for accessing dataset methods."""
        self.parent = parent

    def __call__(self, image, boxes):
        if torch.rand(1) >= self.prob:
            return image, boxes

        assert self.parent is not None, "Parent is not set. MixUp cannot retrieve additional data."

        # Retrieve another image and its boxes randomly from the dataset
        image2, boxes2 = self.parent.get_more_data()[0]

        # Calculate the mixup lambda parameter
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 0.5

        # Mix images
        image1, image2 = TF.to_tensor(image), TF.to_tensor(image2)
        mixed_image = lam * image1 + (1 - lam) * image2

        # Mix bounding boxes
        mixed_boxes = torch.cat([lam * boxes, (1 - lam) * boxes2])

        return TF.to_pil_image(mixed_image), mixed_boxes
