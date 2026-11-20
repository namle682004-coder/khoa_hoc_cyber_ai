# src/lab/config.py

import yaml
from pathlib import Path


class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass


def load_config(path: str) -> dict:
    config_path = Path(path)

    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML format: {e}")

    if not isinstance(config, dict):
        raise ConfigError("Config file must contain a dictionary.")

    # -------- Validate required sections --------
    required_sections = ["data", "model"]

    for section in required_sections:
        if section not in config:
            raise ConfigError(f"Missing required config section: '{section}'")

    # -------- Validate data section --------
    data_cfg = config["data"]

    if "test_size" not in data_cfg:
        raise ConfigError("Missing 'data.test_size' in config")

    if not (0 < data_cfg["test_size"] < 1):
        raise ConfigError("data.test_size must be between 0 and 1")

    # -------- Validate model section --------
    model_cfg = config["model"]

    if "name" not in model_cfg:
        raise ConfigError("Missing 'model.name' in config")

    return config