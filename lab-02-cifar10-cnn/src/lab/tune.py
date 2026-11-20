import json
import itertools
from pathlib import Path
from copy import deepcopy

from src.lab.config import load_config
from src.lab.train import main


CONFIG_PATH = "src/configs/config.yaml"

search_space = {
    "lr": [0.001, 0.0005, 0.0003],
    "batch_size": [64, 128],
    "weight_decay": [0, 1e-4, 5e-4]
}


def run_experiment(base_config, params):

    config_dict = vars(base_config).copy()

    for k, v in params.items():
        config_dict[k] = v

    temp_config = "src/configs/temp_tune.json"

    Path("configs").mkdir(exist_ok=True)

    with open(temp_config, "w") as f:
        json.dump(config_dict, f, indent=2)

    main(temp_config)

    with open("outputs/reports/metrics.json") as f:
        metrics = json.load(f)

    return metrics["best_val_acc"]


def main_tune():

    base_config = load_config(CONFIG_PATH)

    keys = search_space.keys()
    values = search_space.values()

    best_acc = 0
    best_params = None

    for combo in itertools.product(*values):

        params = dict(zip(keys, combo))

        print("\nTesting:", params)

        acc = run_experiment(base_config, params)

        print("Val Acc:", acc)

        if acc > best_acc:
            best_acc = acc
            best_params = params

    print("\nBEST RESULT")

    print(best_params, best_acc)
    Path("outputs").mkdir(exist_ok=True)

    with open("outputs/best_config.json", "w") as f:
        json.dump(
            {
                "best_params": best_params,
                "best_val_acc": best_acc
            },
            f,
            indent=2
        )


if __name__ == "__main__":
    main_tune()