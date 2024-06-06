# YOLO: Official Implementation of YOLOv{7, 9}

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
$pip install git+https://github.com/WongKinYiu/yolov9mit.git
$yolo task=inference task.source=0 # source could be a single file, video, image folder, webcam ID
```

## Introduction
- [**YOLOv9**: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)
- [**YOLOv7**: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors](https://arxiv.org/abs/2207.02696)

## Installation
To get started with YOLOv9, clone this repository and install the required dependencies:
```shell
git clone git@github.com:WongKinYiu/yolov9mit.git
cd yolov9mit
pip install -r requirements.txt
```

## Features
- [x] Autodownload weights/datasets
- [x] Pip installable
- [x] Support for devices:
  - [x] CUDA
  - [x] MPS (PyTorch 2.3+)
  - [x] CPU
- [x] Task:
  - [x] Training
  - [x] Inference
  - [ ] Validation

## Task
These are simple examples. For more customization details, please refer to [Notebooks](examples) and lower-level modifications **[HOWTO](docs/HOWTO)**.

## Training
To train YOLOv9 on your dataset:

1. Modify the configuration file `data/config.yaml` to point to your dataset.
2. Run the training script:
```shell
python lazy.py task=train task.batch_size=8 model=v9-c
```

### Transfer Learning
To perform transfer learning with YOLOv9:
```shell
python lazy.py task=train task.batch_size=8 model=v9-c task.data.dataset={dataset_config}
```

### Inference
To evaluate the model performance, use:
```shell
python lazy.py weights=v9-c.pt # if cloned from GitHub
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
[![Star History Chart](https://api.star-history.com/svg?repos=WongKinYiu/yolov9mit&type=Date)](https://star-history.com/#WongKinYiu/yolov9mit&Date)

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