from dataclasses import dataclass
import yaml


@dataclass
class TrainConfig:
    data_dir: str
    batch_size: int
    num_workers: int
    lr: float
    epochs: int
    output_dir: str
    fast_dev_run: bool = False
    weight_decay: float = 0.0


def load_config(path: str) -> TrainConfig:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    return TrainConfig(**cfg)