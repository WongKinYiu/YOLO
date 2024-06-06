import os
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
from loguru import logger
from omegaconf import ListConfig, OmegaConf

from yolo.config.config import Config, ModelConfig, YOLOLayer
from yolo.tools.dataset_preparation import prepare_weight
from yolo.tools.drawer import draw_model
from yolo.utils.logging_utils import log_model_structure
from yolo.utils.module_utils import get_layer_map


class YOLO(nn.Module):
    """
    A preliminary YOLO (You Only Look Once) model class still under development.

    Parameters:
        model_cfg: Configuration for the YOLO model. Expected to define the layers,
                   parameters, and any other relevant configuration details.
    """

    def __init__(self, model_cfg: ModelConfig):
        super(YOLO, self).__init__()
        self.num_classes = model_cfg.class_num
        self.layer_map = get_layer_map()  # Get the map Dict[str: Module]
        self.model: List[YOLOLayer] = nn.ModuleList()
        self.build_model(model_cfg.model)

    def build_model(self, model_arch: Dict[str, List[Dict[str, Dict[str, Dict]]]]):
        self.layer_index = {}
        output_dim, layer_idx = [3], 1
        logger.info(f"🚜 Building YOLO")
        for arch_name in model_arch:
            logger.info(f"  🏗️  Building {arch_name}")
            for layer_idx, layer_spec in enumerate(model_arch[arch_name], start=layer_idx):
                layer_type, layer_info = next(iter(layer_spec.items()))
                layer_args = layer_info.get("args", {})

                # Get input source
                source = self.get_source_idx(layer_info.get("source", -1), layer_idx)

                # Find in channels
                if any(module in layer_type for module in ["Conv", "ELAN", "ADown", "CBLinear"]):
                    layer_args["in_channels"] = output_dim[source]
                if "Detection" in layer_type:
                    layer_args["in_channels"] = [output_dim[idx] for idx in source]
                    layer_args["num_classes"] = self.num_classes

                # create layers
                layer = self.create_layer(layer_type, source, layer_info, **layer_args)
                self.model.append(layer)

                if layer.tags:
                    if layer.tags in self.layer_index:
                        raise ValueError(f"Duplicate tag '{layer_info['tags']}' found.")
                    self.layer_index[layer.tags] = layer_idx

                out_channels = self.get_out_channels(layer_type, layer_args, output_dim, source)
                output_dim.append(out_channels)
                setattr(layer, "out_c", out_channels)
            layer_idx += 1

    def forward(self, x):
        y = {0: x}
        output = []
        for index, layer in enumerate(self.model, start=1):
            if isinstance(layer.source, list):
                model_input = [y[idx] for idx in layer.source]
            else:
                model_input = y[layer.source]
            x = layer(model_input)
            y[-1] = x
            if layer.usable:
                y[index] = x
            if layer.output:
                output.append(x)
        return output

    def get_out_channels(self, layer_type: str, layer_args: dict, output_dim: list, source: Union[int, list]):
        if any(module in layer_type for module in ["Conv", "ELAN", "ADown"]):
            return layer_args["out_channels"]
        if layer_type == "CBFuse":
            return output_dim[source[-1]]
        if layer_type in ["Pool", "UpSample"]:
            return output_dim[source]
        if layer_type == "Concat":
            return sum(output_dim[idx] for idx in source)
        if layer_type == "IDetect":
            return None

    def get_source_idx(self, source: Union[ListConfig, str, int], layer_idx: int):
        if isinstance(source, ListConfig):
            return [self.get_source_idx(index, layer_idx) for index in source]
        if isinstance(source, str):
            source = self.layer_index[source]
        if source < -1:
            source += layer_idx
        if source > 0:  # Using Previous Layer's Output
            self.model[source - 1].usable = True
        return source

    def create_layer(self, layer_type: str, source: Union[int, list], layer_info: Dict, **kwargs) -> YOLOLayer:
        if layer_type in self.layer_map:
            layer = self.layer_map[layer_type](**kwargs)
            setattr(layer, "layer_type", layer_type)
            setattr(layer, "source", source)
            setattr(layer, "in_c", kwargs.get("in_channels", None))
            setattr(layer, "output", layer_info.get("output", False))
            setattr(layer, "tags", layer_info.get("tags", None))
            setattr(layer, "usable", 0)
            return layer
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")


def create_model(model_cfg: ModelConfig, weight_path: str) -> YOLO:
    """Constructs and returns a model from a Dictionary configuration file.

    Args:
        config_file (dict): The configuration file of the model.

    Returns:
        YOLO: An instance of the model defined by the given configuration.
    """
    OmegaConf.set_struct(model_cfg, False)
    model = YOLO(model_cfg)
    logger.info("✅ Success load model")
    if weight_path:
        if os.path.exists(weight_path):
            model.model.load_state_dict(torch.load(weight_path), strict=False)
            logger.info("✅ Success load model weight")
        else:
            logger.info(f"🌐 Weight {weight_path} not found, try downloading")
            prepare_weight(weight_path=weight_path)

    log_model_structure(model.model)
    draw_model(model=model)
    return model
