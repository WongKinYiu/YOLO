from dataclasses import dataclass
from typing import List, Dict, Union


@dataclass
class Model:
    anchor: List[List[int]]
    model: Dict[str, List[Dict[str, Union[Dict, List, int]]]]


@dataclass
class Download:
    auto: bool
    path: str


@dataclass
class Config:
    model: Model
    download: Download
