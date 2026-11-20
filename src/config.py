"""
src/config.py
Created on October 13, 2026

@author: Yangtingting Wang
https://github.com/xiguashuangjiang
"""

import os
from pathlib import Path
import yaml


def load_config(config_path="config.yaml"):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    api_config = config.setdefault(
        "api",
        {},
    )

    api_url_env = api_config.get(
        "api_url_env",
        "CUSTOM_API_URL",
    )

    api_url = os.getenv(
        api_url_env,
        "",
    ).strip()

    if api_url:
        api_config["custom_api_url"] = api_url

    return config
