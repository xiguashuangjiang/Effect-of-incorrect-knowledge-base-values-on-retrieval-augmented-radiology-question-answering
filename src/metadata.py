"""
src/metadata.py
Created on October 13, 2026

@author: Yangtingting Wang
https://github.com/xiguashuangjiang
"""

import copy
import subprocess
from pathlib import Path
from typing import Any


def get_git_commit(repo_root: str | Path) -> str:

    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "rev-parse",
            "HEAD",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout.strip()


def sanitize_config(config: dict) -> dict:

    clean = copy.deepcopy(config)

    credential_keys = {
        "api_key",
        "apikey",
        "access_token",
        "auth_token",
        "password",
        "client_secret",
    }

    def scrub(value: Any):
        if isinstance(value, dict):
            result = {}

            for key, item in value.items():
                key_lower = key.lower()

                if key_lower.endswith("_env"):
                    result[key] = item

                elif key_lower in credential_keys:
                    result[key] = "[REDACTED]"

                else:
                    result[key] = scrub(item)

            return result

        if isinstance(value, list):
            return [
                scrub(item)
                for item in value
            ]

        return value

    return scrub(clean)
