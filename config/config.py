from dataclasses import dataclass
from typing import Dict, List, Union


@dataclass
class Model:
    anchor: List[List[int]]
    model: Dict[str, List[Dict[str, Union[Dict, List, int]]]]


@dataclass
class Download:
    auto: bool
    path: str


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
class Config:
    model: Model
    download: Download
