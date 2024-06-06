import os
import random
from typing import List, Optional, Union

import numpy as np
import torch
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image


def draw_bboxes(
    img: Union[Image.Image, torch.Tensor],
    bboxes: List[List[Union[int, float]]],
    *,
    save_path: str = "",
    save_name: str = "visualize.png",
    idx2label: Optional[list],
):
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
            logger.warning("🔍 >3 dimension tensor detected, using the 0-idx image.")
            img, bboxes = img[0], bboxes[0]
        img = to_pil_image(img)

    draw = ImageDraw.Draw(img, "RGBA")

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for bbox in bboxes:
        class_id, x_min, y_min, x_max, y_max, *conf = [float(val) for val in bbox]
        bbox = [(x_min, y_min), (x_max, y_max)]

        random.seed(int(class_id))
        color_map = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        draw.rounded_rectangle(bbox, outline=(*color_map, 170), radius=5)
        draw.rounded_rectangle(bbox, fill=(*color_map, 50), radius=5)

        class_text = str(idx2label[int(class_id)] if idx2label else class_id)
        label_text = f"{class_text}" + (f" {conf[0]: .0%}" if conf else "")

        text_bbox = font.getbbox(label_text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = (text_bbox[3] - text_bbox[1]) * 1.25

        text_background = [(x_min, y_min), (x_min + text_width, y_min + text_height)]
        draw.rounded_rectangle(text_background, fill=(*color_map, 175), radius=2)
        draw.text((x_min, y_min), label_text, fill="white", font=font)

    os.makedirs(save_path, exist_ok=True)
    save_image_path = os.path.join(save_path, save_name)
    img.save(save_image_path)  # Save the image with annotations
    logger.info(f"💾 Saved visualize image at {save_image_path}")
    return img


def draw_model(*, model_cfg=None, model=None, v7_base=False):
    from graphviz import Digraph

    if model_cfg:
        from yolo.model.yolo import create_model

        model = create_model(model_cfg)
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
    try:
        dot.render("Model-arch", format="png", cleanup=True)
    except:
        logger.info("Warning: Could not find graphviz backend, continue without drawing the model architecture")
    logger.info("🎨 Drawing Model Architecture at Model-arch.png")
