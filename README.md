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

# Installing Dependencies
```
pip install -r requirements.txt
```

# Training the Model
To start training the model, use the following command:
```bash
$python examples/example_train.py
```
Ensure that you are in the root directory of the project when you run this command.

The training script will automatically utilize the configuration files from the `yolo/config/` directory as specified by the Hydra configuration framework. This setup allows easy management and adjustment of training parameters without modifying the code.

To override any configuration parameters directly from the command line, you can specify them after the script. For example, to change the batch size, you can run:

```bash
$python examples/example_train.py hyper.data.batch_size=8
```



## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=WongKinYiu/yolov9mit&type=Date)](https://star-history.com/#WongKinYiu/yolov9mit&Date)

## To-Do Lists
- [ ] Project Setup
    - [X] requirements
    - [x] LICENSE
    - [ ] README
    - [x] pytests
    - [ ] setup.py/pip install
    - [x] log format
    - [ ] hugging face
- [ ] Data proccess
    - [ ] Dataset
        - [x] Download script
        - [ ] Auto Download
        - [ ] xywh, xxyy, xcyc
    - [x] Dataloder
        - [x] Data arugment
- [ ] Model
    - [ ] load model
        - [ ] from yaml
        - [ ] from github
    - [x] trainer
        - [x] train_one_iter
        - [x] train_one_epoch
    - [ ] DDP
    - [x] EMA, OTA
- [ ] Loss
- [ ] Run
    - [ ] train
    - [ ] test
    - [ ] demo
- [x] Configuration
