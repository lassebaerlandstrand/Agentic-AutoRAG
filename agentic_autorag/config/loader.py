"""YAML configuration loading and validation."""

from pathlib import Path

import yaml

from agentic_autorag.config.models import SearchSpace


def load_config(config_path: str | Path) -> SearchSpace:
    """Load a YAML config file and return a validated SearchSpace.

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

    return SearchSpace.model_validate(raw)
