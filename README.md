# YOLOv9-MIT
An MIT license rewrite of YOLOv9

![WIP](https://img.shields.io/badge/status-WIP-orange)
> [!IMPORTANT]
> This project is currently a Work In Progress and may undergo significant changes. It is not recommended for use in production environments until further notice. Please check back regularly for updates.
> 
> Use of this code is at your own risk and discretion. It is advisable to consult with the project owner before deploying or integrating into any critical systems.

## Contributing

While the project's structure is still being finalized, we ask that potential contributors wait for these foundational decisions to be made. We greatly appreciate your patience and are excited to welcome contributions from the community once we are ready. Alternatively, you are welcome to propose functions that should be implemented based on the original YOLO version or suggest other enhancements! 

If you are interested in contributing, please keep an eye on project updates or contact us directly at [henrytsui000@gmail.com](mailto:henrytsui000@gmail.com) for more information.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=WongKinYiu/yolov9mit&type=Date)](https://star-history.com/#WongKinYiu/yolov9mit&Date)

## To-Do Lists
- [ ] Project Setup
    - [X] requirements
    - [ ] LICENSE
    - [ ] README
    - [ ] pytests
    - [ ] setup.py/pip install
    - [ ] log format
    - [ ] hugging face
- [ ] Data proccess
    - [ ] Dataset
        - [ ] Download script
        - [ ] Auto Download
        - [ ] xywh, xxyy, xcyc
    - [ ] Dataloder
        - [ ] Data augment
- [ ] Model
    - [ ] load model
        - [ ] from yaml
        - [ ] from github
    - [ ] trainer
        - [ ] train_one_iter
        - [ ] train_one_epoch
    - [ ] DDP, EMA, OTA
- [ ] Run
    - [ ] train
    - [ ] test
    - [ ] demo
- [ ] Configuration
    - [ ] hyperparams: dataclass
    - [ ] model cfg: yaml