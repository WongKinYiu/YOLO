import sys
from pathlib import Path

import hydra

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.model.yolo import create_model
from yolo.tools.data_loader import create_dataloader
from yolo.tools.solver import ModelTester, ModelTrainer, ModelValidator
from yolo.utils.bounding_box_utils import Vec2Box
from yolo.utils.deploy_utils import FastModelLoader
from yolo.utils.logging_utils import ProgressLogger
from yolo.utils.model_utils import get_device


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    progress = ProgressLogger(cfg, exp_name=cfg.name)
    device, use_ddp = get_device(cfg.device)
    dataloader = create_dataloader(cfg.task.data, cfg.dataset, cfg.task.task, use_ddp)
    if getattr(cfg.task, "fast_inference", False):
        model = FastModelLoader(cfg).load_model(device)
    else:
        model = create_model(cfg.model, class_num=cfg.class_num, weight_path=cfg.weight)
        model = model.to(device)

    vec2box = Vec2Box(model, cfg.image_size, device)

    if cfg.task.task == "train":
        trainer = ModelTrainer(cfg, model, vec2box, progress, device, use_ddp)
        trainer.solve(dataloader)

    if cfg.task.task == "inference":
        tester = ModelTester(cfg, model, vec2box, progress, device)
        tester.solve(dataloader)

    if cfg.task.task == "validation":
        valider = ModelValidator(cfg.task, model, vec2box, progress, device)
        valider.solve(dataloader)


if __name__ == "__main__":
    main()
