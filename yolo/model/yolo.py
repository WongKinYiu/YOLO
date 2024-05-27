from typing import Any, Dict, List, Union

import torch.nn as nn
from loguru import logger
from omegaconf import ListConfig, OmegaConf

from yolo.config.config import Model
from yolo.tools.layer_helper import get_layer_map


class YOLO(nn.Module):
    """
    A preliminary YOLO (You Only Look Once) model class still under development.

    Parameters:
        model_cfg: Configuration for the YOLO model. Expected to define the layers,
                   parameters, and any other relevant configuration details.
    """

    def __init__(self, model_cfg: Model):
        super(YOLO, self).__init__()
        self.num_classes = model_cfg["num_classes"]
        self.layer_map = get_layer_map()  # Get the map Dict[str: Module]
        self.build_model(model_cfg.model)

    def build_model(self, model_arch: Dict[str, List[Dict[str, Dict[str, Dict]]]]):
        model_list = nn.ModuleList()
        output_dim = [3]
        layer_indices_by_tag = {}
        layer_idx = 1
        logger.info(f"🚜 Building YOLO")
        for arch_name in model_arch:
            logger.info(f"  🏗️  Building {arch_name}")
            for layer_idx, layer_spec in enumerate(model_arch[arch_name], start=layer_idx):
                layer_type, layer_info = next(iter(layer_spec.items()))
                layer_args = layer_info.get("args", {})

                # Get input source
                source = layer_info.get("source", -1)
                if isinstance(source, str):
                    source = layer_indices_by_tag[source]
                elif isinstance(source, ListConfig):
                    source = [layer_indices_by_tag[idx] if isinstance(idx, str) else idx for idx in source]

                # Find in channels
                if any(module in layer_type for module in ["Conv", "ELAN", "ADown", "CBLinear"]):
                    layer_args["in_channels"] = output_dim[source]
                if "Detection" in layer_type:
                    layer_args["in_channels"] = [output_dim[idx] for idx in source]
                    layer_args["num_classes"] = self.num_classes

                # create layers
                layer = self.create_layer(layer_type, source, layer_info, **layer_args)
                model_list.append(layer)

                if "tags" in layer_info:
                    if layer_info["tags"] in layer_indices_by_tag:
                        raise ValueError(f"Duplicate tag '{layer_info['tags']}' found.")
                    layer_indices_by_tag[layer_info["tags"]] = layer_idx

                out_channels = self.get_out_channels(layer_type, layer_args, output_dim, source)
                output_dim.append(out_channels)
            layer_idx += 1

        self.model = model_list

    def forward(self, x):
        y = [x]
        output = []
        for layer in self.model:
            if isinstance(layer.source, list):
                model_input = [y[idx] for idx in layer.source]
            else:
                model_input = y[layer.source]
            x = layer(model_input)
            y.append(x)
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

    def create_layer(self, layer_type: str, source: Union[int, list], layer_info, **kwargs):
        if layer_type in self.layer_map:
            layer = self.layer_map[layer_type](**kwargs)
            setattr(layer, "source", source)
            setattr(layer, "output", layer_info.get("output", False))
            setattr(layer, "tags", layer_info.get("tags", None))
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
