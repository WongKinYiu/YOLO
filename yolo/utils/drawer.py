from typing import List, Union

import torch
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image


def draw_bboxes(img: Union[Image.Image, torch.Tensor], bboxes: List[List[Union[int, float]]]):
    """
    Draw bounding boxes on an image.

    Args:
    - img (PIL Image or torch.Tensor): Image on which to draw the bounding boxes.
    - bboxes (List of Lists/Tensors): Bounding boxes with [class_id, x_min, y_min, x_max, y_max],
      where coordinates are normalized [0, 1].
    """
    # Convert tensor image to PIL Image if necessary
    if isinstance(img, torch.Tensor):
        if img.dim() > 3:
            logger.info("Multi-frame tensor detected, using the first image.")
            img = img[0]
            bboxes = bboxes[0]
        img = to_pil_image(img)

    draw = ImageDraw.Draw(img)
    width, height = img.size
    font = ImageFont.load_default(30)

    for bbox in bboxes:
        class_id, x_min, y_min, x_max, y_max = bbox
        x_min = x_min * width
        x_max = x_max * width
        y_min = y_min * height
        y_max = y_max * height
        shape = [(x_min, y_min), (x_max, y_max)]
        draw.rectangle(shape, outline="red", width=3)
        draw.text((x_min, y_min), str(int(class_id)), font=font, fill="blue")

    img.save("visualize.jpg")  # Save the image with annotations
    logger.info("Saved visualize image at visualize.png")
