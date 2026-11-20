"""
rar_pipeline/configuration.py
Created on October 13, 2026

@author: Yangtingting Wang
https://github.com/xiguashuangjiang
"""

import os
from enum import Enum
from dataclasses import dataclass, fields
from typing import Any, Optional, Dict, get_origin, get_args

from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv

load_dotenv()


class SearchAPI(Enum):
    SEARXNG = "searxng"
    LOCAL = "local"


def _coerce_field_value(field_type: Any, raw_value: Any) -> Any:
    if raw_value is None:
        return None

    origin = get_origin(field_type)
    args = get_args(field_type)

    if origin is Optional and args:
        return _coerce_field_value(args[0], raw_value)

    if isinstance(field_type, type) and issubclass(field_type, Enum):
        if isinstance(raw_value, field_type):
            return raw_value
        if isinstance(raw_value, str):
            try:
                return field_type(raw_value)
            except ValueError:
                try:
                    return field_type[raw_value]
                except (KeyError, ValueError):
                    pass
        return raw_value

    if field_type is int:
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            return raw_value
    if field_type is float:
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return raw_value
    if field_type is bool:
        if isinstance(raw_value, str):
            return raw_value.lower() in ("1", "true", "yes", "on")
        return bool(raw_value)

    return raw_value


@dataclass(kw_only=True)
class Configuration:

    number_of_queries: int = 4
    max_search_depth: int = 2
    temperature: float = 0
    max_tokens: int = 1536
    recursion_limit: int = 30
    search_api: SearchAPI = SearchAPI.LOCAL

    supervisor_model: str = os.getenv("SUPERVISOR_MODEL", "openai:gpt-4o-mini")
    researcher_model: str = os.getenv("RESEARCHER_MODEL", "openai:gpt-4o-mini")

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        raw_values: dict[str, Any] = {}
        for f in fields(cls):
            if not f.init:
                continue
            env_val = os.environ.get(f.name.upper(), None)
            cfg_val = configurable.get(f.name)
            chosen = env_val if env_val is not None else cfg_val
            coerced = _coerce_field_value(f.type, chosen)
            if coerced is not None and coerced != "":
                raw_values[f.name] = coerced

        return cls(**raw_values)
