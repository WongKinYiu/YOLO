from typing import List, Union

import numpy as np
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


def draw_model(*, model_cfg=None, model=None, v7_base=False):
    from graphviz import Digraph

    if model_cfg:
        from yolo.model.yolo import get_model

        model = get_model(model_cfg)
    elif model is None:
        raise ValueError("Drawing Object is None")

    model_size = len(model.model) + 1
    model_mat = np.zeros((model_size, model_size), dtype=bool)

    layer_name = ["INPUT"]
    for idx, layer in enumerate(model.model, start=1):
        layer_name.append(str(type(layer)).split(".")[-1][:-2])
        if layer.tags is not None:
            layer_name[-1] = f"{layer.tags}-{layer_name[-1]}"
        if isinstance(layer.source, int):
            source = layer.source + (layer.source < 0) * idx
            model_mat[source, idx] = True
        else:
            for source in layer.source:
                source = source + (source < 0) * idx
                model_mat[source, idx] = True

    pattern_mat = []
    if v7_base:
        pattern_list = [("ELAN", 8, 3), ("ELAN", 8, 55), ("MP", 5, 11)]
        for name, size, position in pattern_list:
            pattern_mat.append(
                (name, size, model_mat[position : position + size, position + 1 : position + 1 + size].copy())
            )

    dot = Digraph(comment="Model Flow Chart")
    node_idx = 0

    for idx in range(model_size):
        for jdx in range(idx, model_size - 7):
            for name, size, pattern in pattern_mat:
                if (model_mat[idx : idx + size, jdx : jdx + size] == pattern).all():
                    layer_name[idx] = name
                    model_mat[idx : idx + size, jdx : jdx + size] = False
                    model_mat[idx, idx + size] = True
        dot.node(str(idx), f"{layer_name[idx]}")
        node_idx += 1
        for jdx in range(idx, model_size):
            if model_mat[idx, jdx]:
                dot.edge(str(idx), str(jdx))

    dot.render("Model-arch", format="png", cleanup=True)
    logger.info("🎨 Drawing Model Architecture at Model-arch.png")
