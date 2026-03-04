"""YAML configuration loading and validation."""

from pathlib import Path

import yaml

from agentic_autorag.config.models import ProjectConfig


def load_config(config_path: str | Path) -> ProjectConfig:
    """Load a YAML config file and return a validated ProjectConfig.

    Pydantic's model_validate handles all structural and type validation.
    Raises FileNotFoundError if the file doesn't exist, or
    pydantic.ValidationError if the YAML content is invalid.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Config file is empty: {path}")

    return ProjectConfig.model_validate(raw)
