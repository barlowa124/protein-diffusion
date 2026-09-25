"""Shared config loader. DIFFUSION_CONFIG env var overrides the default."""

import os
from pathlib import Path

import yaml


def load_config() -> dict:
    path = Path(os.environ.get("DIFFUSION_CONFIG", "config/config.yaml"))
    with open(path) as f:
        return yaml.safe_load(f)
