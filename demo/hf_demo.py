import sys
from pathlib import Path

import gradio
import torch
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).resolve().parent.parent))

from yolo import (
    AugmentationComposer,
    NMSConfig,
    Vec2Box,
    bbox_nms,
    create_model,
    draw_bboxes,
)

DEFAULT_MODEL = "v9-c"
IMAGE_SIZE = (640, 640)


def load_model(model_name, device):
    model_cfg = OmegaConf.load(f"yolo/config/model/{model_name}.yaml")
    model_cfg.model.auxiliary = {}
    model = create_model(model_cfg, True)
    model.to(device).eval()
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(DEFAULT_MODEL, device)
v2b = Vec2Box(model, IMAGE_SIZE, device)
class_list = OmegaConf.load("yolo/config/general.yaml").class_list

transform = AugmentationComposer([])


def predict(model_name, image, nms_confidence, nms_iou):
    global DEFAULT_MODEL, model, device, v2b, class_list
    if model_name != DEFAULT_MODEL:
        model = load_model(model_name, device)
        v2b = Vec2Box(model, IMAGE_SIZE, device)
        DEFAULT_MODEL = model_name

    image_tensor, _, rev_tensor = transform(image)

    image_tensor = image_tensor.to(device)[None]
    rev_tensor = rev_tensor.to(device)

    with torch.no_grad():
        predict = model(image_tensor)
        pred_class, _, pred_bbox = v2b(predict["Main"])

    nms_config = NMSConfig(nms_confidence, nms_iou)

    pred_bbox = pred_bbox / rev_tensor[0] - rev_tensor[None, None, 1:]
    pred_bbox = bbox_nms(pred_class, pred_bbox, nms_config)
    result_image = draw_bboxes(image, pred_bbox, idx2label=class_list)

    return result_image


interface = gradio.Interface(
    fn=predict,
    inputs=[
        gradio.components.Dropdown(choices=["v9-c", "v9-m", "v9-s"], value="v9-c", label="Model Name"),
        gradio.components.Image(type="pil", label="Input Image"),
        gradio.components.Slider(0, 1, step=0.01, value=0.5, label="NMS Confidence Threshold"),
        gradio.components.Slider(0, 1, step=0.01, value=0.5, label="NMS IoU Threshold"),
    ],
    outputs=gradio.components.Image(type="pil", label="Output Image"),
)

if __name__ == "__main__":
    interface.launch()
