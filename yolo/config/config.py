from dataclasses import dataclass
from typing import Dict, List, Union

from torch import nn


@dataclass
class AnchorConfig:
    reg_max: int
    strides: List[int]


@dataclass
class Model:
    anchor: AnchorConfig
    model: Dict[str, List[Dict[str, Union[Dict, List, int]]]]


@dataclass
class Download:
    auto: bool
    path: str


@dataclass
class DataLoaderConfig:
    batch_size: int
    class_num: int
    image_size: List[int]
    shuffle: bool
    pin_memory: bool


@dataclass
class OptimizerArgs:
    lr: float
    weight_decay: float


@dataclass
class OptimizerConfig:
    type: str
    args: OptimizerArgs


@dataclass
class SchedulerArgs:
    step_size: int
    gamma: float


@dataclass
class SchedulerConfig:
    type: str
    args: SchedulerArgs
    warmup: Dict[str, Union[str, int, float]]


@dataclass
class EMAConfig:
    enabled: bool
    decay: float


@dataclass
class MatcherConfig:
    iou: str
    topk: int
    factor: Dict[str, int]


@dataclass
class LossConfig:
    objective: List[List]
    aux: Union[bool, float]
    matcher: MatcherConfig


@dataclass
class TrainConfig:
    epoch: int
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    ema: EMAConfig
    loss: LossConfig


@dataclass
class GeneralConfig:
    out_path: str
    task: str
    device: Union[str, int, List[int]]
    cpu_num: int
    use_wandb: bool
    lucky_number: 10
    exist_ok: bool
    resume_train: bool
    use_TensorBoard: bool


@dataclass
class HyperConfig:
    general: GeneralConfig
    data: DataLoaderConfig
    train: TrainConfig


@dataclass
class Dataset:
    file_name: str
    num_files: int


@dataclass
class Datasets:
    base_url: str
    images: Dict[str, Dataset]


@dataclass
class Download:
    auto: bool
    save_path: str
    datasets: Datasets


@dataclass
class YOLOLayer(nn.Module):
    source: Union[int, str, List[int]]
    output: bool
    tags: str
    layer_type: str
    usable: bool

    def __post_init__(self):
        super().__init__()


@dataclass
class Config:
    model: Model
    download: Download
    hyper: HyperConfig
    name: str
