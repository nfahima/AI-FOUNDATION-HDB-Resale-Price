"""
Validates the structure and types of the pipeline's configuration dictionary.

This is a small, dependency-free schema check rather than a full library
(e.g. pydantic/jsonschema) so it stays easy to read for a course project,
but it catches the same class of bug: missing keys, wrong types, and
silently-wrong values (like test_size being a list instead of a float).
"""

import os
from typing import Any, Dict, List


REQUIRED_KEYS = {
    "file_path": str,
    "numerical_features": list,
    "nominal_features": list,
    "ordinal_features": list,
    "flat_type_categories": list,
    "passthrough_features": list,
    "target_column": str,
    "test_size": (int, float),
    "random_state": int,
    "val_size": (int, float),
    "param_grid": dict,
    "cv": int,
    "scoring": str,
    "n_jobs": int,
}


class ConfigError(ValueError):
    """Raised when the configuration file is missing keys or has bad types."""


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validates that all required keys are present and correctly typed.

    Args:
        config: The configuration dictionary loaded from YAML.

    Raises:
        ConfigError: If a required key is missing, has the wrong type,
            or holds a value outside its valid range.
    """
    missing: List[str] = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ConfigError(f"Missing required config keys: {missing}")

    wrong_type: List[str] = []
    for key, expected_type in REQUIRED_KEYS.items():
        if not isinstance(config[key], expected_type):
            wrong_type.append(
                f"'{key}' should be {expected_type}, got {type(config[key]).__name__}"
            )
    if wrong_type:
        raise ConfigError("Config type errors: " + "; ".join(wrong_type))

    if not 0 < config["test_size"] < 1:
        raise ConfigError(f"test_size must be between 0 and 1, got {config['test_size']}")
    if not 0 < config["val_size"] < 1:
        raise ConfigError(f"val_size must be between 0 and 1, got {config['val_size']}")
    if not os.path.exists(config["file_path"]):
        raise ConfigError(f"file_path does not exist: {config['file_path']}")