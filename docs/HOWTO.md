# How To modified YOLO

To facilitate easy customization of the YOLO model, we've structured the codebase to allow for changes through configuration files and minimal code adjustments. This guide will walk you through the steps to customize various components of the model including the architecture, blocks, data loaders, and loss functions.

## Custom Model Architecture

You can change the model architecture simply by modifying the YAML configuration file. Here's how:

1. **Modify Architecture in Config:**
   
   Navigate to your model's configuration file (typically formate like `yolo/config/model/v9-c.yaml`).
   - Adjust the architecture settings under the `architecture` section. Ensure that every module you reference exists in `module.py`, or refer to the next section on how to add new modules.

    ```yaml
    model:
      foo:
        - ADown:
            args: {out_channels: 256}
        - RepNCSPELAN:
            source: -2
            args: {out_channels: 512, part_channels: 256}
            tags: B4
      bar:
        - Concat:
            source: [-2, B4]
    ```

   `tags`: Use this to labels any module you want, and could be the module source.

   `source`: Set this to the index of the module output you wish to use as input; default is `-1` which refers to the last module's output. Capable tags, relative position, absolute position

   `args`: A dictionary used to initialize parameters for convolutional or bottleneck layers.

   `output`: Whether to serve as the output of the model.

## Custom Block

To add or modify a block in the model:

1. **Create a New Module:**

   Define a new class in `module.py` that inherits from `nn.Module`.
   
   The constructor should accept `in_channels` as a parameter. Make sure to calculate `out_channels` based on your model's requirements or configure it through the YAML file using `args`.

    ```python
    class CustomBlock(nn.Module):
        def __init__(self, in_channels, out_channels, **kwargs):
            super().__init__()
            self.module = # conv, bool, ...
        def forward(self, x):
            return self.module(x) 
    ```

2. **Reference in Config:**
   ```yaml
    ...
    - CustomBlock:
        args: {out_channels: int, etc: ...}
        ...
    ...
   ```


## Custom Data Augmentation

Custom transformations should be designed to accept an image and its bounding boxes, and return them after applying the desired changes. Here’s how you can define such a transformation:


1. **Define Dataset:**

    Your class must have a `__call__` method that takes a PIL image and its corresponding bounding boxes as input, and returns them after processing.


   ```python
    class CustomTransform:
        def __init__(self, prob=0.5):
            self.prob = prob

        def __call__(self, image, boxes):
            return image, boxes
   ```
2. **Update CustomTransform in Config:**

    Specify your custom transformation in a YAML config `yolo/config/data/augment.yaml`. For examples:
    ```yaml
    Mosaic: 1
    # ... (Other Transform)
    CustomTransform: 0.5
    ```
