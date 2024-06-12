# YOLO: Official Implementation of YOLOv9, YOLOv7

![WIP](https://img.shields.io/badge/status-WIP-orange)
> [!IMPORTANT]
> This project is currently a Work In Progress and may undergo significant changes. It is not recommended for use in production environments until further notice. Please check back regularly for updates.
>
> Use of this code is at your own risk and discretion. It is advisable to consult with the project owner before deploying or integrating into any critical systems.

Welcome to the official implementation of YOLOv7 and YOLOv9. This repository will contains the complete codebase, pre-trained models, and detailed instructions for training and deploying YOLOv9.

## TL;DR
- This is the official YOLO model implementation with an MIT License.
- For quick deployment: you can enter directly in the terminal:
```shell
pip install git+git@github.com:WongKinYiu/YOLO.git
yolo task=inference task.source=0 # source could be a single file, video, image folder, webcam ID
```

## Introduction
- [**YOLOv9**: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)
- [**YOLOv7**: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors](https://arxiv.org/abs/2207.02696)

## Installation
To get started with YOLOv9, clone this repository and install the required dependencies:
```shell
git clone git@github.com:WongKinYiu/YOLO.git
cd YOLO
pip install -r requirements.txt
```

## Features

<table>
<tr><td>

| Tools | pip 🐍 | HuggingFace 🤗 | Docker 🐳 |
| -------------------- | :----: | :--------------: | :-------: |
| Compatibility       | ✅     | ❔               | 🧪        |

|  Phase    | Training | Validation | Inference |
| ------------------- | :------: | :---------: | :-------: |
| Supported           | ✅       | ✅          | ✅        |

</td><td>

| Device | CUDA       | CPU       | MPS       |
| ------------------ | :---------: | :-------: | :-------: |
| PyTorch            | v1.12      | v2.3+     | v1.12     |
| ONNX               | ✅         | ✅        | -         |
| TensorRT           | ✅         | -        | -         |
| OpenVINO           | -          | 🧪        | ❔        |

</td></tr> </table>



## Task
These are simple examples. For more customization details, please refer to [Notebooks](examples) and lower-level modifications **[HOWTO](docs/HOWTO.md)**.

## Training
To train YOLO on your dataset:

1. Modify the configuration file `data/config.yaml` to point to your dataset.
2. Run the training script:
```shell
python yolo/lazy.py task=train task.data.batch_size=8 model=v9-c
```

### Transfer Learning
To perform transfer learning with YOLOv9:
```shell
python yolo/lazy.py task=train task.data.batch_size=8 model=v9-c dataset={dataset_config} device={cpu, mps, cuda}
```

### Inference
To evaluate the model performance, use:
```shell
python yolo/lazy.py task=inference weight=weights/v9-c.pt model=v9-c task.fast_inference=deploy # use deploy weight
python yolo/lazy.py task=inference # if cloned from GitHub
yolo task=inference task.data.source={Any} # if pip installed
```

### Validation [WIP]
To validate the model performance, use:
```shell
# Work In Progress...
```

## Contributing
Contributions to the YOLOv9 project are welcome! See [CONTRIBUTING](docs/CONTRIBUTING.md) for guidelines on how to contribute.

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=WongKinYiu/YOLO&type=Date)](https://star-history.com/#WongKinYiu/YOLO&Date)

## Citations
```
@misc{wang2024yolov9,
      title={YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information},
      author={Chien-Yao Wang and I-Hau Yeh and Hong-Yuan Mark Liao},
      year={2024},
      eprint={2402.13616},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
