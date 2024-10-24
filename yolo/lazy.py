import sys
from pathlib import Path

import hydra
from lightning import Trainer

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.tools.solver import TrainModel, ValidateModel
from yolo.utils.logging_utils import setup


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    callbacks, loggers = setup(cfg)

    trainer = Trainer(
        accelerator="cuda",
        max_epochs=getattr(cfg.task, "epoch", None),
        precision="16-mixed",
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=1,
        gradient_clip_val=10,
        deterministic=True,
    )

    match cfg.task.task:
        case "train":
            model = TrainModel(cfg)
            trainer.fit(model)
        case "validation":
            model = ValidateModel(cfg)
            trainer.validate(model)


if __name__ == "__main__":
    main()
