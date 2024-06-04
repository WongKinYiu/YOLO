import sys
from pathlib import Path

import hydra
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.model.yolo import create_model
from yolo.tools.data_loader import create_dataloader
from yolo.tools.solver import ModelTester, ModelTrainer
from yolo.utils.logging_utils import custom_logger, validate_log_directory


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    custom_logger()
    save_path = validate_log_directory(cfg, cfg.name)
    dataloader = create_dataloader(cfg)
    device = torch.device(cfg.device)
    model = create_model(cfg).to(device)

    if cfg.task.task == "train":
        trainer = ModelTrainer(cfg, model, save_path, device)
        trainer.solve(dataloader)

    if cfg.task.task == "inference":
        tester = ModelTester(cfg, model, save_path, device)
        tester.solve(dataloader)


if __name__ == "__main__":
    main()
