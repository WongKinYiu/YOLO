from __future__ import annotations
import dataclasses
import pathlib
from PIL import Image
import torch

@dataclasses.dataclass
class YoloImage:
    image_path: pathlib.Path | None = None
    bboxes: list[BoundingBox] = dataclasses.field(default_factory=list)
    
    @property
    def image(self):
        return Image.open(self.image_path).convert("RGB")
    
    
@dataclasses.dataclass
class BoundingBox:
    class_id: int
    x: float # center_x
    y: float # center_y
    width: float 
    height: float

    @property
    def tensor(self):
        """
        Tensor representation of the bounding box.
        Format: <class_id> <x_min> <y_min> <x_max> <y_max>
            
        """
        return torch.tensor([self.class_id, self.x - self.width / 2, self.y - self.height / 2, self.x + self.width / 2, self.y + self.height / 2])