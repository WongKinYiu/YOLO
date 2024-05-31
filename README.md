# YOLO: Official Implementation of YOLOv{7, 9}

![WIP](https://img.shields.io/badge/status-WIP-orange)
> [!IMPORTANT]
> This project is currently a Work In Progress and may undergo significant changes. It is not recommended for use in production environments until further notice. Please check back regularly for updates.
> 
> Use of this code is at your own risk and discretion. It is advisable to consult with the project owner before deploying or integrating into any critical systems.

Welcome to the official implementation of the YOLOv7 and YOLOv9. This repository will contains the complete codebase, pre-trained models, and detailed instructions for training and deploying YOLOv9.

## TL;DR
- Official YOLOv9 model implementation.
- Features real-time detection with state-of-the-art accuracy.
<!-- - Includes pre-trained models and training scripts. -->
- Quick train: `python examples/example_train.py`

## Introduction
- [**YOLOv9**: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)
- [**YOLOv7**: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)

## Installation
To get started with YOLOv9, clone this repository and install the required dependencies:
```shell
git clone git@github.com:WongKinYiu/yolov9mit.git
cd yolov9mit
pip install -r requirements.txt
```

<!-- 
```
pip install git+https://github.com/WongKinYiu/yolov9mit.git
``` 
-->

<!-- ### Quick Start
Run YOLOv9 on a pre-trained model with:

```shell
python examples/example_train.py hyper.data.batch_size=8
``` -->

<!-- ## Model Zoo[WIP]
Find pre-trained models with benchmarks on various datasets in the [Model Zoo](docs/MODELS). -->

## Training
For training YOLOv9 on your dataset:

Modify the configuration file data/config.yaml to point to your dataset.
Run the training script:

```shell
python examples/example_train.py hyper.data.batch_size=8 model=v9-c
```

More customization details, or ways to modify the model can be found [HOWTO](docs/HOWTO).

## Evaluation [WIP]
Evaluate the model performance using:

```shell
python examples/examples_evaluate.py weights=v9-c.pt
```

## Contributing
Contributions to the YOLOv9 project are welcome! See [CONTRIBUTING](docs/CONTRIBUTING.md) for how to help out.

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