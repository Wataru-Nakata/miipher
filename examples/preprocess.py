import hydra
from omegaconf import DictConfig
from lightning.pytorch import seed_everything

from miipher.preprocess.preprocessor import Preprocessor


@hydra.main(version_base="1.3", config_name="config", config_path="./configs")
def main(cfg: DictConfig):
    seed_everything(1234)
    preprocssor = Preprocessor(cfg=cfg)
    preprocssor.build_from_path()


if __name__ == "__main__":
    main()
