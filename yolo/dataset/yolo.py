from typing import  List
import pathlib
from rich.progress import track

from yolo.dataset.base import BaseDataset
from yolo.config.config import DatasetConfig
from yolo.utils import primitives

 
class YoloDataset(BaseDataset):
    """
    YOLO Dataset Format
    **Directory Structure**

    ```
    dataset/
    ├── images/
    │   ├── subset_1/
    │   │   ├── image1.jpg
    │   │   └── ...
    │   └── subset_2/
    │       ├── image2.jpg
    │       └── ...
    └── labels/
    ├── subset_1/
    │   ├── image1.txt
    │   └── ...
    └── subset_2/
        ├── image2.txt
        └── ...
    ```

    **Annotation Format**

    Each annotation file corresponds to an image and contains one or more lines:

    ```
    <class_id> <center_x> <center_y> <width> <height>
    ```

    - **class_id**: Zero-based index matching a line in `classes.txt`.
    - **center_x**, **center_y**: Normalized coordinates of the bounding box center (values between 0 and 1).
    - **width**, **height**: Normalized dimensions of the bounding box (values between 0 and 1).

    **Normalization**

    Coordinates are normalized relative to image dimensions:

    ```
    center_x = bbox_center_x / image_width
    center_y = bbox_center_y / image_height
    width = bbox_width / image_width
    height = bbox_height / image_height
    ```

    **Notes**
    - Image and annotation filenames must match exactly (excluding extensions).
    - The coordinate system origin `(0,0)` is at the top-left corner of the image.
    - If an image contains no objects, its annotation file should be empty or omitted.
    - Supported image formats include `.jpg`, `.jpeg`, and `.png`.
    - Ensure consistency between training and validation splits.

    """

    def __init__(self, root_path:str, subset:str, **kwargs):
        self.root_path = pathlib.Path(root_path)
        self.subset = subset
        super().__init__(**kwargs)
        

    def extract_data(self) -> List[primitives.YoloImage]:
        """ Prepare the data for the dataset.
        Args:
            dataset_cfg: The dataset configuration.
            
        Returns:
            List[Image]: The list of images.
        """
        images = []
        for image_path in track((self.root_path / "images" / self.subset).glob("*.*"), description="Loading images"):
            if not image_path.suffix in (".jpg", ".jpeg", ".png"):
                continue
            image = primitives.YoloImage(image_path=image_path)
            image.bboxes = self.load_bounding_boxes(self.root_path / "labels" / self.subset / (image_path.stem + ".txt"))
            images.append(image)
        return images
            
            
    def load_bounding_boxes(self, path:pathlib.Path) -> List[primitives.BoundingBox]:
        boxes = []
        if not path.exists():
            return boxes
        with open(path, "r") as file:
            for line in file:
                elements = line.strip().split()
                class_id = int(elements[0])
                center_x, center_y, width, height = map(float, elements[1:])                
                boxes.append(primitives.BoundingBox(class_id=class_id, x=center_x, y=center_y, width=width, height=height))
                
        return boxes