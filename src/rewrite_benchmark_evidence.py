"""
src/rewrite_benchmark_evidence.py
Created on October 22, 2026

@author: Yangtingting Wang
https://github.com/xiguashuangjiang
"""

import asyncio
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from langchain_openai import ChatOpenAI

from src.hashing import sha256_json, sha256_text
from src.metadata import get_git_commit, sanitize_config
from src.reports import load_json
from src.config import load_config


STYLE_DESCRIPTIONS = {
    "forum": (
        "an informal post in an online forum, for example: "
        "hey so basically this condition is when..."
    ),
    "overconfident": (
        "absolute language, for example: this definitively proves, "
        "always indicates, there is no doubt"
    ),
    "hedged": (
        "maximal uncertainty, for example: may possibly suggest, "
        "some authors have speculated, the evidence is inconclusive"
    ),
    "review": (
        "a formal, authoritative systematic review"
    ),
}

NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")

REWRITE_PROMPT = """Rewrite the following medical text in the style described below.
Style: {style}
Keep every medical statement, every number, every unit, and every negation of the original text. Do not add any medical content and do not remove any.
Reply with the rewritten text only.

Text:
{evidence_clean}"""

CHECK_PROMPT = """Original text:
{evidence_clean}

Rewritten text:
{rewrite}

Does the rewritten text contain the same medical claims as the original text, with nothing added and nothing removed? Answer with one word: YES, NO, or PARTIALLY."""


def _load_evidence_clean_map(source_csv: str) -> dict[tuple[str, int], str]:
    path = Path(source_csv)

    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark CSV not found: {path}"
        )

    evidence_map: dict[tuple[str, int], str] = {}

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise RuntimeError(
                f"No header found in benchmark CSV: {path}"
            )

        required = {
            "dataset",
            "q_number",
            "evidence_clean",
        }

        missing = required - set(reader.fieldnames)

        if missing:
            raise KeyError(
                "Benchmark CSV is missing required column(s): "
                f"{sorted(missing)}. "
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:
            dataset = str(
                row["dataset"]
            ).strip()

            q_number = int(
                str(
                    row["q_number"]
                ).strip()
            )

            evidence = row.get(
                "evidence_clean",
                "",
            )

            if evidence is None:
                evidence = ""

            evidence_map[
                (
                    dataset,
                    q_number,
                )
            ] = str(
                evidence
            )

    return evidence_map


def _attach_evidence_clean(
    questions: list[dict],
    config: dict,
) -> list[dict]:
    evidence_map = _load_evidence_clean_map(
        config[
            "benchmark"
        ][
            "source_csv"
        ]
    )

    enriched = []

    for question in questions:
        dataset = str(
            question[
                "dataset"
            ]
        ).strip()

        q_number = int(
            question[
                "q_number"
            ]
        )

        key = (
            dataset,
            q_number,
        )

        if key not in evidence_map:
            raise KeyError(
                "No evidence_clean found in source CSV for "
                f"{dataset} q{q_number}"
            )

        copied = dict(
            question
        )

        copied[
            "evidence_clean"
        ] = evidence_map[
            key
        ]

        enriched.append(
            copied
        )

    empty = [
        (
            q["dataset"],
            q["q_number"],
        )
        for q in enriched
        if not str(
            q.get(
                "evidence_clean",
                "",
            )
        ).strip()
    ]

    if empty:
        raise ValueError(
            "evidence_clean is empty for "
            f"{len(empty)} benchmark row(s), "
            f"e.g. {empty[:10]}"
        )

    return enriched

def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    tmp.replace(path)


def _load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a list in {path}"
        )

    return data


def _record_key(
    dataset: str,
    q_number: int,
    style: str,
) -> tuple[str, int, str]:
    return (
        str(dataset),
        int(q_number),
        str(style),
    )


def _model_name_for_api(name: str) -> str:
    if name.startswith("custom:"):
        return name.split(":", 1)[1]

    return name


def _build_model(
    config: dict,
    model_name: str,
    temperature: float,
    max_tokens: int,
    extra_body: dict | None = None,
):
    return ChatOpenAI(
        base_url=config["api"]["custom_api_url"].rstrip("/"),
        api_key=os.getenv(
            config["api"]["api_key_env"],
            "sk-dummy-key",
        ),
        model=_model_name_for_api(
            model_name
        ),
        temperature=float(
            temperature
        ),
        max_tokens=int(
            max_tokens
        ),
        timeout=float(
            config[
                "style_rewrite"
            ][
                "timeout_seconds"
            ]
        ),
        max_retries=0,
        streaming=False,
        extra_body=extra_body,
        http_client=httpx.Client(
            verify=False
        ),
        http_async_client=httpx.AsyncClient(
            verify=False
        ),
    )


def _response_text(response) -> str:
    content = getattr(
        response,
        "content",
        "",
    )

    if isinstance(
        content,
        str,
    ):
        return content.strip()

    return str(
        content
    ).strip()


def _response_metadata(
    response,
) -> dict:
    response_metadata = (
        getattr(
            response,
            "response_metadata",
            {},
        )
        or {}
    )

    usage_metadata = (
        getattr(
            response,
            "usage_metadata",
            {},
        )
        or {}
    )

    token_usage = (
        response_metadata.get(
            "token_usage",
            {},
        )
        or {}
    )

    prompt_tokens = (
        usage_metadata.get(
            "input_tokens"
        )
    )

    if prompt_tokens is None:
        prompt_tokens = (
            token_usage.get(
                "prompt_tokens"
            )
        )

    completion_tokens = (
        usage_metadata.get(
            "output_tokens"
        )
    )

    if completion_tokens is None:
        completion_tokens = (
            token_usage.get(
                "completion_tokens"
            )
        )

    return {
        "model_served":
            response_metadata.get(
                "model_name"
            )
            or response_metadata.get(
                "model"
            )
            or "",

        "prompt_tokens":
            prompt_tokens,

        "completion_tokens":
            completion_tokens,

        "finish_reason":
            response_metadata.get(
                "finish_reason",
                "",
            ),
    }


def _numbers_check(
    original: str,
    rewrite: str,
) -> bool:
    original_numbers = Counter(
        NUMBER_RE.findall(
            original
        )
    )

    rewrite_numbers = Counter(
        NUMBER_RE.findall(
            rewrite
        )
    )

    return (
        original_numbers
        == rewrite_numbers
    )


def _parse_checker_label(
    checker_text: str,
) -> str:
    match = re.search(
        r"\b(YES|NO|PARTIALLY)\b",
        checker_text.upper(),
    )

    if not match:
        return "UNPARSED"

    return match.group(1)


def _resume_sha256(
    question: dict,
    style: str,
    config: dict,
) -> str:
    return sha256_json({
        "dataset":
            str(
                question[
                    "dataset"
                ]
            ),

        "q_number":
            int(
                question[
                    "q_number"
                ]
            ),

        "evidence_clean":
            question.get(
                "evidence_clean",
                "",
            ),

        "style":
            style,

        "style_description":
            STYLE_DESCRIPTIONS[
                style
            ],

        "rewrite_prompt":
            REWRITE_PROMPT,

        "checker_prompt":
            CHECK_PROMPT,

        "rewrite_model":
            config[
                "models"
            ][
                "report_model"
            ],

        "checker_model":
            config[
                "models"
            ][
                "checker_model"
            ],

        "checker_chat_template_kwargs":
            {
                "enable_thinking": False,
            },

        "style_rewrite":
            config[
                "style_rewrite"
            ],
    })


async def _rewrite_one(
    config: dict,
    question: dict,
    style: str,
) -> dict:
    original = question.get(
        "evidence_clean",
        "",
    )

    if not isinstance(
        original,
        str,
    ):
        original = str(
            original or ""
        )

    if not original.strip():
        raise ValueError(
            "evidence_clean is empty"
        )

    max_attempts = int(
        config[
            "style_rewrite"
        ][
            "max_attempts"
        ]
    )

    first_temperature = float(
        config[
            "style_rewrite"
        ][
            "temperature_first_attempt"
        ]
    )

    retry_temperature = float(
        config[
            "style_rewrite"
        ][
            "temperature_retry"
        ]
    )

    rewrite_max_tokens = int(
        config[
            "style_rewrite"
        ][
            "rewrite_max_tokens"
        ]
    )

    checker_max_tokens = int(
        config[
            "style_rewrite"
        ][
            "checker_max_tokens"
        ]
    )

    last_rewrite = ""
    last_numbers_check = "fail"
    last_content_check = "UNPARSED"
    last_error = None

    rewrite_calls = []
    checker_calls = []

    for attempt in range(
        1,
        max_attempts + 1,
    ):
        temperature = (
            first_temperature
            if attempt == 1
            else retry_temperature
        )


        try:
            rewrite_model = _build_model(
                config=config,
                model_name=config[
                    "models"
                ][
                    "report_model"
                ],
                temperature=temperature,
                max_tokens=rewrite_max_tokens,
            )

            rewrite_prompt = (
                REWRITE_PROMPT.format(
                    style=
                        STYLE_DESCRIPTIONS[
                            style
                        ],
                    evidence_clean=
                        original,
                )
            )

            rewrite_response = (
                await
                rewrite_model.ainvoke([
                    {
                        "role":
                            "user",

                        "content":
                            rewrite_prompt,
                    }
                ])
            )

            rewrite = _response_text(
                rewrite_response
            )

            rewrite_meta = (
                _response_metadata(
                    rewrite_response
                )
            )

            rewrite_meta.update({
                "attempt":
                    attempt,

                "temperature":
                    temperature,
            })

            rewrite_calls.append(
                rewrite_meta
            )

            last_rewrite = rewrite

            numbers_ok = (
                _numbers_check(
                    original,
                    rewrite,
                )
            )

            last_numbers_check = (
                "pass"
                if numbers_ok
                else "fail"
            )

            checker_model = (
                _build_model(
                    config=config,
                    model_name=config[
                        "models"
                    ][
                        "checker_model"
                    ],
                    temperature=0,
                    max_tokens=
                        max(
                            checker_max_tokens,
                            256,
                        ),
                    extra_body={
                        "chat_template_kwargs": {
                            "enable_thinking": False,
                        },
                    },
                )
            )

            checker_prompt = (
                CHECK_PROMPT.format(
                    evidence_clean=
                        original,
                    rewrite=
                        rewrite,
                )
            )

            checker_response = (
                await
                checker_model.ainvoke([
                    {
                        "role":
                            "user",

                        "content":
                            checker_prompt,
                    }
                ])
            )

            checker_text = (
                _response_text(
                    checker_response
                )
            )

            checker_meta = (
                _response_metadata(
                    checker_response
                )
            )

            checker_meta.update({
                "attempt":
                    attempt,

                "response_text":
                    checker_text,
            })

            checker_calls.append(
                checker_meta
            )

            content_label = (
                _parse_checker_label(
                    checker_text
                )
            )

            last_content_check = (
                content_label
            )


            if (
                numbers_ok
                and content_label
                == "YES"
            ):
                return {
                    "dataset":
                        str(
                            question[
                                "dataset"
                            ]
                        ),

                    "q_number":
                        int(
                            question[
                                "q_number"
                            ]
                        ),

                    "style":
                        style,

                    "text":
                        rewrite,

                    "text_sha256":
                        sha256_text(
                            rewrite
                        ),

                    "attempts":
                        attempt,

                    "numbers_check":
                        "pass",

                    "content_check":
                        "YES",

                    "status":
                        "accepted",

                    "rewrite_model_calls":
                        rewrite_calls,

                    "checker_model_calls":
                        checker_calls,

                    "error":
                        None,

                    "resume_sha256":
                        _resume_sha256(
                            question,
                            style,
                            config,
                        ),
                }

            last_error = (
                "QC failed: "
                f"numbers="
                f"{last_numbers_check}, "
                f"content="
                f"{last_content_check}"
            )

        except Exception as exc:
            last_error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )


    return {
        "dataset":
            str(
                question[
                    "dataset"
                ]
            ),

        "q_number":
            int(
                question[
                    "q_number"
                ]
            ),

        "style":
            style,

        "text":
            last_rewrite,

        "text_sha256":
            sha256_text(
                last_rewrite
            ),

        "attempts":
            max_attempts,

        "numbers_check":
            last_numbers_check,

        "content_check":
            last_content_check,

        "status":
            "failed",

        "rewrite_model_calls":
            rewrite_calls,

        "checker_model_calls":
            checker_calls,

        "error":
            last_error,

        "resume_sha256":
            _resume_sha256(
                question,
                style,
                config,
            ),
    }


def _served_models_from_records(
    records: list[dict],
) -> dict:
    rewrite_served = set()
    checker_served = set()

    for record in records:
        for call in record.get(
            "rewrite_model_calls",
            [],
        ):
            model = call.get(
                "model_served"
            )

            if model:
                rewrite_served.add(
                    model
                )

        for call in record.get(
            "checker_model_calls",
            [],
        ):
            model = call.get(
                "model_served"
            )

            if model:
                checker_served.add(
                    model
                )

    return {
        "rewrite_model_served":
            sorted(
                rewrite_served
            ),

        "checker_model_served":
            sorted(
                checker_served
            ),
    }


async def run_style_rewrite(
    config: dict,
) -> Path:
    questions = load_json(
        config[
            "benchmark"
        ][
            "runtime_json"
        ]
    )

    if not isinstance(
        questions,
        list,
    ):
        raise RuntimeError(
            "Benchmark runtime JSON "
            "must contain a list."
        )

    if len(questions) != 200:
        raise RuntimeError(
            f"Expected 200 questions, "
            f"got {len(questions)}"
        )

    questions = _attach_evidence_clean(
        questions,
        config,
    )

    styles = list(
        config[
            "style_rewrite"
        ][
            "styles"
        ]
    )

    unknown_styles = [
        style
        for style in styles
        if style
        not in STYLE_DESCRIPTIONS
    ]

    if unknown_styles:
        raise ValueError(
            "Unknown style(s): "
            f"{unknown_styles}"
        )

    selected = questions
    output_filename = (
        "ev_styles.json"
    )
    settings_filename = (
        "ev_styles.json"
    )

    output_path = (
        Path(
            config[
                "outputs"
            ][
                "corruption_direct_evidence"
            ]
        )
        / "texts"
        / output_filename
    )

    settings_path = (
        Path(
            config[
                "outputs"
            ][
                "corruption_direct_evidence"
            ]
        )
        / "settings"
        / settings_filename
    )

    existing_list = (
        _load_existing(
            output_path
        )
    )

    existing = {
        _record_key(
            record[
                "dataset"
            ],
            record[
                "q_number"
            ],
            record[
                "style"
            ],
        ):
            record
        for record
        in existing_list
    }

    settings_record = {
        "run_type":
            "style_rewrite",

        "n_questions":
            len(selected),

        "n_styles":
            len(styles),

        "git_commit":
            get_git_commit(
                config[
                    "project"
                ][
                    "repo_root"
                ]
            ),

        "config":
            sanitize_config(
                config
            ),

        "rewrite_model_requested":
            config[
                "models"
            ][
                "report_model"
            ],

        "checker_model_requested":
            config[
                "models"
            ][
                "checker_model"
            ],

        "checker_chat_template_kwargs":
            {
                "enable_thinking": False,
            },

        "rewrite_prompt":
            REWRITE_PROMPT,

        "checker_prompt":
            CHECK_PROMPT,

        "style_descriptions":
            STYLE_DESCRIPTIONS,

        "decoding":
            {
                "rewrite_temperature_first_attempt":
                    config[
                        "style_rewrite"
                    ][
                        "temperature_first_attempt"
                    ],

                "rewrite_temperature_retry":
                    config[
                        "style_rewrite"
                    ][
                        "temperature_retry"
                    ],

                "rewrite_max_tokens":
                    config[
                        "style_rewrite"
                    ][
                        "rewrite_max_tokens"
                    ],

                "checker_temperature":
                    0,

                "checker_max_tokens_configured":
                    config[
                        "style_rewrite"
                    ][
                        "checker_max_tokens"
                    ],

                "checker_max_tokens_effective":
                    max(
                        int(
                            config[
                                "style_rewrite"
                            ][
                                "checker_max_tokens"
                            ]
                        ),
                        256,
                    ),

                "max_attempts":
                    config[
                        "style_rewrite"
                    ][
                        "max_attempts"
                    ],

                "timeout_seconds":
                    config[
                        "style_rewrite"
                    ][
                        "timeout_seconds"
                    ],
            },

        "served_models_observed":
            {
                "rewrite_model_served":
                    [],

                "checker_model_served":
                    [],
            },
    }

    _save_json(
        settings_path,
        settings_record,
    )


    counter = 0


    for question in selected:
        for style in styles:
            counter += 1

            key = _record_key(
                question[
                    "dataset"
                ],
                question[
                    "q_number"
                ],
                style,
            )

            current_hash = (
                _resume_sha256(
                    question,
                    style,
                    config,
                )
            )

            old = existing.get(
                key
            )

            if (
                old is not None
                and old.get(
                    "resume_sha256"
                )
                == current_hash
                and old.get(
                    "status"
                )
                in {
                    "accepted",
                    "failed",
                }
            ):

                continue


            record = (
                await
                _rewrite_one(
                    config,
                    question,
                    style,
                )
            )

            existing[
                key
            ] = record

            ordered = []

            for q in selected:
                for s in styles:
                    k = _record_key(
                        q[
                            "dataset"
                        ],
                        q[
                            "q_number"
                        ],
                        s,
                    )

                    if k in existing:
                        ordered.append(
                            existing[
                                k
                            ]
                        )

            _save_json(
                output_path,
                ordered,
            )

            served = (
                _served_models_from_records(
                    ordered
                )
            )

            settings_record[
                "served_models_observed"
            ] = served

            _save_json(
                settings_path,
                settings_record,
            )


    return output_path


def main_tone_rewrites(config_path):
    return asyncio.run(run_style_rewrite(load_config(config_path)))
