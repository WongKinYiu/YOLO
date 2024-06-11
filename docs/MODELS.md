# YOLO Model Zoo

Welcome to the YOLOv9 Model Zoo! Here, you will find a variety of pre-trained models tailored to different use cases and performance needs. Each model comes with detailed information about its training regime, performance metrics, and usage instructions.

## Standard Models

These models are trained on common datasets like COCO and provide a balance between speed and accuracy.


| Model | Support? |Test Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> | Param. | FLOPs |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| [**YOLOv9-S**]() |✅ | 640  | **46.8%** | **63.4%** | **50.7%** | **7.1M** | **26.4G** |
| [**YOLOv9-M**]() |✅ | 640  | **51.4%** | **68.1%** | **56.1%** | **20.0M** | **76.3G** |
| [**YOLOv9-C**]() |✅ | 640  | **53.0%** | **70.2%** | **57.8%** | **25.3M** | **102.1G** |
| [**YOLOv9-E**]() | 🔧 | 640  | **55.6%** | **72.8%** | **60.6%** | **57.3M** | **189.0G** |
|  |  |  |  |  |  |  |
| [**YOLOv7**]() |🔧 | 640  | **51.4%** | **69.7%** | **55.9%** |
| [**YOLOv7-X**]() |🔧 | 640  | **53.1%** | **71.2%** | **57.8%** |
| [**YOLOv7-W6**]() | 🔧 | 1280 | **54.9%** | **72.6%** | **60.1%** |
| [**YOLOv7-E6**]() | 🔧 | 1280 | **56.0%** | **73.5%** | **61.2%** |
| [**YOLOv7-D6**]() | 🔧 | 1280 | **56.6%** | **74.0%** | **61.8%** |
| [**YOLOv7-E6E**]() | 🔧 | 1280 | **56.8%** | **74.4%** | **62.1%** |

## Download and Usage Instructions

To use these models, download them from the links provided and use the following command to run detection:

```bash
$yolo detect weights=path/to/model.pt img=640 conf=0.25 source=your_image.jpg
```
