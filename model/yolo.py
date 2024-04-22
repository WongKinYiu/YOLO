from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
from loguru import logger
from omegaconf import OmegaConf
from tools.layer_helper import get_layer_map


class YOLO(nn.Module):
    """
    A preliminary YOLO (You Only Look Once) model class still under development.

    Parameters:
        model_cfg: Configuration for the YOLO model. Expected to define the layers,
                   parameters, and any other relevant configuration details.
    """

    def __init__(self, model_cfg: Dict[str, Any]):
        super(YOLO, self).__init__()
        self.nc = model_cfg["nc"]
        self.layer_map = get_layer_map()  # Get the map Dict[str: Module]
        self.build_model(model_cfg.model)

    def build_model(self, model_arch: Dict[str, List[Dict[str, Dict[str, Dict]]]]):
        model_list = nn.ModuleList()
        output_dim = [3]
        layer_indices_by_tag = {}
        for arch_name in model_arch:
            logger.info(f"🏗️  Building model-{arch_name}")
            for layer_idx, layer_spec in enumerate(model_arch[arch_name], start=1):
                layer_type, layer_info = next(iter(layer_spec.items()))
                layer_args = layer_info.get("args", {})
                source = layer_info.get("source", -1)

                if isinstance(source, str):
                    source = layer_indices_by_tag[source]
                if "Conv" in layer_type:
                    layer_args["in_channels"] = output_dim[source]
                if "Detect" in layer_type:
                    layer_args["nc"] = self.nc
                    layer_args["ch"] = [output_dim[idx] for idx in source]

                layer = self.create_layer(layer_type, source, **layer_args)
                model_list.append(layer)

                if "tags" in layer_info:
                    if layer_info["tags"] in layer_indices_by_tag:
                        raise ValueError(f"Duplicate tag '{layer_info['tags']}' found.")
                    layer_indices_by_tag[layer_info["tags"]] = layer_idx

                out_channels = self.get_out_channels(layer_type, layer_args, output_dim, source)
                output_dim.append(out_channels)
        self.model = model_list

    def forward(self, x):
        y = [x]
        for layer in self.model:
            if OmegaConf.is_list(layer.source):
                model_input = [y[idx] for idx in layer.source]
            else:
                model_input = y[layer.source]
            x = layer(model_input)
            y.append(x)
        return x

    def get_out_channels(self, layer_type: str, layer_args: dict, output_dim: list, source: Union[int, list]):
        if "Conv" in layer_type:
            return layer_args["out_channels"]
        if layer_type in ["MaxPool", "UpSample"]:
            return output_dim[source]
        if layer_type == "Concat":
            return sum(output_dim[idx] for idx in source)
        if layer_type == "IDetect":
            return None

    def create_layer(self, layer_type: str, source: Union[int, list], **kwargs):
        if layer_type in self.layer_map:
            layer = self.layer_map[layer_type](**kwargs)
            layer.source = source
            return layer
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")


def get_model(model_cfg: dict) -> YOLO:
    """Constructs and returns a model from a Dictionary configuration file.

    Args:
        config_file (dict): The configuration file of the model.

    Returns:
        YOLO: An instance of the model defined by the given configuration.
    """
    OmegaConf.set_struct(model_cfg, False)
    model = YOLO(model_cfg)
    logger.info("✅ Success load model")
    return model


if __name__ == "__main__":
    model_cfg = load_model_cfg("v7-base")

    YOLO(model_cfg)
