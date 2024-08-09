import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


class AugmentationComposer:
    """Composes several transforms together."""

    def __init__(self, transforms, image_size: int = [640, 640]):
        self.transforms = transforms
        # TODO: handle List of image_size [640, 640]
        self.image_size = image_size
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


class RemoveOutliers:
    """Removes outlier bounding boxes that are too small or have invalid dimensions."""

    def __init__(self, min_box_area=1e-8):
        """
        Args:
            min_box_area (float): Minimum area for a box to be kept, as a fraction of the image area.
        """
        self.min_box_area = min_box_area

    def __call__(self, image, boxes):
        """
        Args:
            image (PIL.Image): The cropped image.
            boxes (torch.Tensor): Bounding boxes in normalized coordinates (x_min, y_min, x_max, y_max).
        Returns:
            PIL.Image: The input image (unchanged).
            torch.Tensor: Filtered bounding boxes.
        """
        box_areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])

        valid_boxes = (box_areas > self.min_box_area) & (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 4] > boxes[:, 2])

        return image, boxes[valid_boxes]


class PadAndResize:
    def __init__(self, image_size, background_color=(114, 114, 114)):
        """Initialize the object with the target image size."""
        self.target_width, self.target_height = image_size
        self.background_color = background_color

    def __call__(self, image: Image, boxes):
        img_width, img_height = image.size
        scale = min(self.target_width / img_width, self.target_height / img_height)
        new_width, new_height = int(img_width * scale), int(img_height * scale)

        resized_image = image.resize((new_width, new_height), Image.LANCZOS)

        pad_left = (self.target_width - new_width) // 2
        pad_top = (self.target_height - new_height) // 2
        padded_image = Image.new("RGB", (self.target_width, self.target_height), self.background_color)
        padded_image.paste(resized_image, (pad_left, pad_top))

        boxes[:, [1, 3]] = (boxes[:, [1, 3]] * new_width + pad_left) / self.target_width
        boxes[:, [2, 4]] = (boxes[:, [2, 4]] * new_height + pad_top) / self.target_height

        transform_info = torch.tensor([scale, pad_left, pad_top, pad_left, pad_top])
        return padded_image, boxes, transform_info


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

        img_sz = self.parent.image_size[0]  # Assuming `image_size` is defined in parent
        more_data = self.parent.get_more_data(3)  # get 3 more images randomly

        data = [(image, boxes)] + more_data
        mosaic_image = Image.new("RGB", (2 * img_sz, 2 * img_sz), (114, 114, 114))
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


class RandomCrop:
    """Randomly crops the image to half its size along with adjusting the bounding boxes."""

    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): Probability of applying the crop.
        """
        self.prob = prob

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            original_width, original_height = image.size
            crop_height, crop_width = original_height // 2, original_width // 2
            top = torch.randint(0, original_height - crop_height + 1, (1,)).item()
            left = torch.randint(0, original_width - crop_width + 1, (1,)).item()

            image = TF.crop(image, top, left, crop_height, crop_width)

            boxes[:, [1, 3]] = boxes[:, [1, 3]] * original_width - left
            boxes[:, [2, 4]] = boxes[:, [2, 4]] * original_height - top

            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, crop_width)
            boxes[:, [2, 4]] = boxes[:, [2, 4]].clamp(0, crop_height)

            boxes[:, [1, 3]] /= crop_width
            boxes[:, [2, 4]] /= crop_height

        return image, boxes
