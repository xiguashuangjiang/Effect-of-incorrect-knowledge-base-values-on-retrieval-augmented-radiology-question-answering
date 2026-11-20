"""
src/evaluate_answers.py
Created on October 22, 2026

@author: Yangtingting Wang
https://github.com/xiguashuangjiang
"""

import asyncio
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import httpx
from langchain_openai import ChatOpenAI

from src.config import load_config
from src.metadata import get_git_commit, sanitize_config
from src.parsing import parse_model_response


ANSWER_PROMPT = """Use the question, the options, and the evidence below to choose the single best answer.

Evidence:
{evidence}

Question:
{question}

Options:
{options}

Choose exactly one option from {option_range}. Then rate your confidence that your chosen option is correct, as a number from 0 to 100.
Reply with exactly these two lines and nothing else:
FINAL_ANSWER: <LETTER>
CONFIDENCE: <NUMBER>"""

CLOSEDBOOK_PROMPT = """Use the question and the options below to choose the single best answer.

Question:
{question}

Options:
{options}

Choose exactly one option from {option_range}. Then rate your confidence that your chosen option is correct, as a number from 0 to 100.
Reply with exactly these two lines and nothing else:
FINAL_ANSWER: <LETTER>
CONFIDENCE: <NUMBER>"""


REPORT_CORRUPTION_DIRECT_EVIDENCE = [
    "clean", "num010", "num025", "num050", "num100",
    "cov010", "cov025", "cov050", "cov100",
]

STYLE_CONDITIONS = {
    "ev_forum": "forum",
    "ev_overconfident": "overconfident",
    "ev_hedged": "hedged",
    "ev_review": "review",
}

DIRECT_CONDITIONS = ["ev_clean", "ev_conflict", "ev_numeric"]
DIRECT_RAG_CONDITIONS = ["rag_clean", "rag_num100"]
REPEAT_NO_RETRIEVAL_CLASSIFICATION_REPORTS = ["clean_r2", "clean_r3", "noretrieval"]

VOLUME_CONDITIONS = [
    "topk01_pad00", "topk03_pad00", "topk05_pad00",
    "topk10_pad00", "topk20_pad00",
    "topk05_pad50", "topk05_pad75", "topk05_pad90", "topk05_pad95",
]

REPORT_MODELS_SOURCE_ABLATION_REPORTS = [
    "top1_100",
    "dsv4_clean",
    "dsv4_clean_r2",
    "dsv4_num100",
    "dsv4_cov100",
    "dsv4_noretrieval",
    "q35_clean",
    "q35_clean_r2",
    "q35_num100",
    "q35_cov100",
    "q35_noretrieval",
]

ALL_CONDITIONS = (
    REPORT_CORRUPTION_DIRECT_EVIDENCE
    + ["closedbook"]
    + DIRECT_CONDITIONS
    + DIRECT_RAG_CONDITIONS
    + list(STYLE_CONDITIONS)
    + REPEAT_NO_RETRIEVAL_CLASSIFICATION_REPORTS
    + VOLUME_CONDITIONS
    + REPORT_MODELS_SOURCE_ABLATION_REPORTS
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(
    path: Path,
    data: Any,
) -> None:

    import time

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

        f.flush()

        try:
            os.fsync(
                f.fileno()
            )
        except OSError:
            pass

    max_replace_attempts = 10

    for attempt in range(
        1,
        max_replace_attempts + 1,
    ):
        try:
            os.replace(
                tmp,
                path,
            )

            return

        except PermissionError:

            if (
                attempt
                == max_replace_attempts
            ):
                raise


            time.sleep(
                1.0
            )

def _question_key(item: dict) -> tuple[str, int]:
    return str(item["dataset"]), int(item["q_number"])


def _requested_model(item: dict) -> str:
    value = item.get("model_requested", item.get("model", ""))
    return str(value)


def _answer_key(item: dict) -> tuple[str, int, str, str]:
    return (
        str(item["dataset"]),
        int(item["q_number"]),
        str(item["condition"]),
        _requested_model(item),
    )


def _extract_question_text(item: dict) -> str:
    for key in ("question", "Question", "question_text", "query", "summary"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise KeyError(f"Could not find question text. Available keys: {sorted(item.keys())}")


def _extract_options(item: dict) -> tuple[str, list[str]]:
    options = None
    for key in ("options", "Options", "answer_options", "choices"):
        if key in item:
            options = item[key]
            break

    labels, lines = [], []

    if isinstance(options, dict):
        for raw_label, text in options.items():
            label = str(raw_label).strip().upper()
            if label.startswith("OPTION "):
                label = label.split()[-1]
            if len(label) == 1:
                labels.append(label)
                lines.append(f"{label}. {text}")

    elif isinstance(options, list):
        for i, value in enumerate(options):
            label = chr(ord("A") + i)
            if isinstance(value, dict):
                text = (
                    value.get("text") or value.get("option")
                    or value.get("content") or value.get("value")
                    or str(value)
                )
            else:
                text = str(value)
            labels.append(label)
            lines.append(f"{label}. {text}")

    if len(labels) not in {4, 5}:
        raise ValueError(
            f"Expected 4 or 5 answer options, got {len(labels)} for "
            f"{item.get('dataset')} q{item.get('q_number')}"
        )

    return "\n".join(lines), labels


def _response_text(response) -> str:
    content = getattr(response, "content", "")
    return content.strip() if isinstance(content, str) else str(content).strip()


def _reasoning_text(response) -> str:
    additional = getattr(response, "additional_kwargs", {}) or {}
    rm = getattr(response, "response_metadata", {}) or {}

    for source in (additional, rm):
        for key in ("reasoning_content", "reasoning_text", "reasoning"):
            value = source.get(key)
            if value is not None:
                return str(value)
    return ""


def _response_meta(response) -> dict:
    rm = getattr(response, "response_metadata", {}) or {}
    um = getattr(response, "usage_metadata", {}) or {}
    tu = rm.get("token_usage", {}) or {}
    prompt_tokens = um.get("input_tokens")
    if prompt_tokens is None:
        prompt_tokens = tu.get("prompt_tokens")
    completion_tokens = um.get("output_tokens")
    if completion_tokens is None:
        completion_tokens = tu.get("completion_tokens")
    return {
        "model_served": rm.get("model_name") or rm.get("model") or "",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": rm.get("finish_reason", ""),
    }


def _parse_answer(
    text: str,
    allowed_labels: list[str],
) -> dict:

    parsed = parse_model_response(
        text
    )

    final_answer = (
        parsed.answer_letter
    )

    if (
        final_answer is not None
        and final_answer not in allowed_labels
    ):
        final_answer = None

    return {
        "parse_status": (
            "valid"
            if final_answer is not None
            else "unparsed"
        ),
        "final_answer": final_answer,
        "confidence": parsed.confidence,
    }

def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _option_range(labels: list[str]) -> str:
    if len(labels) == 4:
        return "A, B, C, or D"
    if len(labels) == 5:
        return "A, B, C, D, or E"
    raise ValueError(f"Expected 4 or 5 labels, got {labels}")


def _build_model(config: dict, model_name: str):
    kwargs = {}

    model_overrides = (
        config.get("answering", {})
        .get("model_overrides", {})
        .get(model_name, {})
    )

    enable_thinking = model_overrides.get("enable_thinking")

    if enable_thinking is None and model_name in {
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
    }:
        enable_thinking = False

    if enable_thinking is not None:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": bool(enable_thinking)
            }
        }

    return ChatOpenAI(
        base_url=config["api"]["custom_api_url"].rstrip("/"),
        api_key=os.getenv(config["api"]["api_key_env"], "sk-dummy-key"),
        model=model_name,
        temperature=float(config["answering"]["temperature"]),
        max_tokens=int(config["answering"]["max_tokens"]),
        timeout=600,
        max_retries=0,
        streaming=False,
        http_client=httpx.Client(verify=False),
        http_async_client=httpx.AsyncClient(verify=False),
        **kwargs,
    )


def resolve_answer_model(config: dict, model_selector: str | int) -> str:
    models = list(config["models"]["answering"])
    if isinstance(model_selector, int):
        index = model_selector
    else:
        selector = str(model_selector).strip()
        if not selector.isdigit():
            return selector
        index = int(selector)

    if index < 1 or index > len(models):
        raise ValueError(f"Model index must be 1-{len(models)}, got {index}")
    return models[index - 1]


def _model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")


def _corruption_direct_evidence_root(config: dict) -> Path:
    return Path(config["outputs"]["corruption_direct_evidence"])


def _repeat_no_retrieval_classification_root(config: dict) -> Path:
    return Path(config["outputs"]["repeat_no_retrieval_classification"])


def _report_models_source_ablation_root(config: dict) -> Path:
    configured = config.get("outputs", {}).get("report_models_source_ablation")
    if configured:
        return Path(configured)

    return Path(config["paths"]["output_root"]) / "report_models_source_ablation"


def _outputs_root(config: dict) -> Path:
    try:
        return Path(os.path.commonpath([str(_corruption_direct_evidence_root(config)), str(_repeat_no_retrieval_classification_root(config))]))
    except Exception:
        return _corruption_direct_evidence_root(config).parent


def _normalize_json(data: Any) -> list[dict]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("records", "reports", "results", "items"):
            if isinstance(data.get(key), list):
                return [x for x in data[key] if isinstance(x, dict)]
        out = []
        for key, value in data.items():
            if isinstance(value, dict):
                row = dict(value)
                if "q_number" not in row and "question_id" not in row:
                    row["question_id"] = key
                out.append(row)
        return out
    return []


def _record_q_number(record: dict) -> int | None:
    for key in ("q_number", "question_number", "question_id", "id"):
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, int):
            return value
        text = str(value).strip()
        if text.isdigit():
            return int(text)
        m = re.search(r"(\d+)$", text)
        if m:
            return int(m.group(1))
    return None


def _record_dataset(record: dict) -> str | None:
    value = record.get("dataset")
    return str(value) if value is not None and str(value).strip() else None


def _report_text(record: dict | None) -> str:
    if not record:
        return ""
    for key in ("report", "deep_research_report", "rar_report", "content", "text"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _direct_text(record: dict | None) -> str:
    if not record:
        return ""
    for key in ("text", "evidence", "evidence_text", "content", "passage"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _report_is_failed(record: dict | None) -> bool:
    if record is None:
        return True
    status = str(record.get("report_status", record.get("status", ""))).strip().lower()
    if status in {"failed", "error", "api_failed", "incomplete"}:
        return True
    return not bool(_report_text(record))


def _condition_family(condition: str) -> str:
    if condition == "closedbook":
        return "closedbook"
    if condition in STYLE_CONDITIONS:
        return "style"
    if condition in DIRECT_CONDITIONS:
        return "direct"
    if condition in DIRECT_RAG_CONDITIONS:
        return "direct_rag"
    if (
        condition in REPORT_CORRUPTION_DIRECT_EVIDENCE
        or condition in REPEAT_NO_RETRIEVAL_CLASSIFICATION_REPORTS
        or condition in VOLUME_CONDITIONS
        or condition in REPORT_MODELS_SOURCE_ABLATION_REPORTS
    ):
        return "report"
    raise ValueError(f"Unknown condition: {condition}")


def _discover_report_file(config: dict, condition: str) -> Path:
    if _condition_family(condition) != "report":
        raise ValueError(f"'{condition}' is not a report condition.")

    report_models_source_ablation_model_conditions = {
        "dsv4_clean",
        "dsv4_clean_r2",
        "dsv4_num100",
        "dsv4_cov100",
        "dsv4_noretrieval",
        "q35_clean",
        "q35_clean_r2",
        "q35_num100",
        "q35_cov100",
        "q35_noretrieval",
    }

    if condition in report_models_source_ablation_model_conditions:
        path = _report_models_source_ablation_root(config) / "reports" / f"{condition}.json"
    else:
        reports_root = Path(
            config.get("outputs", {}).get(
                "formal_reports",
                Path(config["paths"]["output_root"]) / "reports",
            )
        )
        path = reports_root / f"{condition}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Formal report file does not exist for '{condition}':\\n"
            f"{path}\\n"
            "Expected a full 200-question report file."
        )

    rows = _normalize_json(_load_json(path))
    question_keys = {
        (_record_dataset(row), _record_q_number(row))
        for row in rows
        if _record_q_number(row) is not None
    }

    if len(question_keys) != 200:
        raise RuntimeError(
            f"Expected 200 unique report question keys for '{condition}', "
            f"got {len(question_keys)}.\\nFile: {path}"
        )

    return path

def _load_lookup(path: Path) -> dict[tuple[str | None, int], dict]:
    out = {}
    for row in _normalize_json(_load_json(path)):
        q = _record_q_number(row)
        if q is not None:
            out[(_record_dataset(row), q)] = row
    return out


def _lookup_record(lookup: dict, dataset: str, q_number: int) -> dict | None:
    if (dataset, q_number) in lookup:
        return lookup[(dataset, q_number)]
    if (None, q_number) in lookup:
        return lookup[(None, q_number)]
    matches = [v for (ds, q), v in lookup.items() if q == q_number]
    return matches[0] if len(matches) == 1 else None


def _load_style_lookup(config: dict):
    basename = "ev_styles.json"
    path = _corruption_direct_evidence_root(config) / "texts" / basename
    if not path.exists():
        raise FileNotFoundError(f"Could not find {basename} under {_corruption_direct_evidence_root(config)}")

    lookup = {}
    for row in _normalize_json(_load_json(path)):
        if {"dataset", "q_number", "style"}.issubset(row):
            lookup[(str(row["dataset"]), int(row["q_number"]), str(row["style"]).lower())] = row
    return path, lookup


def _norm_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _load_csv_direct_lookup(config: dict) -> tuple[Path, dict[tuple[str, int], dict], dict[str, str]]:
    path = Path(config["benchmark"]["source_csv"])
    if not path.exists():
        raise FileNotFoundError(f"Benchmark source CSV not found:\n{path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        raw_headers = reader.fieldnames or []
        normalized = {_norm_header(h): h for h in raw_headers}

        def pick(candidates: list[str], required: bool = True) -> str | None:
            for candidate in candidates:
                key = _norm_header(candidate)
                if key in normalized:
                    return normalized[key]
            if required:
                raise KeyError(
                    "Could not resolve required CSV column.\n"
                    f"Tried aliases: {candidates}\n"
                    f"Available columns: {raw_headers}"
                )
            return None

        dataset_col = pick(["dataset"])
        q_col = pick(["q_number", "q number", "question_number", "question number"])

        clean_col = pick([
            "evidence_clean",
            "clean_evidence",
            "evidence clean",
            "clean evidence",
        ])

        conflict_col = pick([
            "evidence_conflict",
            "conflict_evidence",
            "evidence conflict",
            "conflict evidence",
            "evidence_contradictory",
            "contradictory_evidence",
            "evidence contradiction",
        ])

        lookup: dict[tuple[str, int], dict] = {}
        for row in reader:
            dataset = str(row.get(dataset_col, "")).strip()
            q_raw = str(row.get(q_col, "")).strip()
            if not dataset or not q_raw:
                continue
            try:
                q_number = int(float(q_raw))
            except ValueError:
                continue

            lookup[(dataset, q_number)] = {
                "ev_clean": str(row.get(clean_col, "") or "").strip(),
                "ev_conflict": str(row.get(conflict_col, "") or "").strip(),
            }


    return path, lookup, {
        "ev_clean": clean_col,
        "ev_conflict": conflict_col,
    }

def _discover_ev_numeric_file(config: dict) -> Path:
    path = (
        Path(config["outputs"]["corruption_direct_evidence"])
        / "texts"
        / "ev_numeric.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"ev_numeric file does not exist:\n{path}"
        )

    return path


def _load_ev_numeric_lookup(
    path: Path,
) -> dict[tuple[str, int], dict]:
    rows = _normalize_json(_load_json(path))

    if len(rows) != 200:
        raise RuntimeError(
            f"Expected 200 ev_numeric records, got {len(rows)}.\n"
            f"File: {path}"
        )

    expected_fields = {
        "dataset",
        "q_number",
        "text",
        "text_sha256",
        "n_values_changed",
    }

    lookup: dict[tuple[str, int], dict] = {}

    for row in rows:
        missing = expected_fields - set(row)
        if missing:
            raise RuntimeError(
                "ev_numeric record is missing required field(s) "
                f"{sorted(missing)}: {row.get('dataset')} q{row.get('q_number')}"
            )

        dataset = str(row["dataset"]).strip()
        q_number = int(row["q_number"])
        evidence = str(row["text"])

        if not evidence.strip():
            raise RuntimeError(
                f"Empty ev_numeric text for {dataset} q{q_number}"
            )

        expected_hash = _sha256_text(evidence)
        stored_hash = str(row["text_sha256"]).strip().lower()
        if stored_hash != expected_hash:
            raise RuntimeError(
                "ev_numeric text_sha256 mismatch for "
                f"{dataset} q{q_number}: stored={stored_hash}, expected={expected_hash}"
            )

        key = (dataset, q_number)
        if key in lookup:
            raise RuntimeError(
                f"Duplicate ev_numeric record: {dataset} q{q_number}"
            )

        lookup[key] = {
            "text": evidence,
            "text_sha256": stored_hash,
            "n_values_changed": int(row["n_values_changed"]),
        }

    if len(lookup) != 200:
        raise RuntimeError(
            "ev_numeric lookup does not contain "
            f"200 unique question IDs: {len(lookup)}"
        )

    return lookup


def _direct_rag_evidence_path(config: dict, condition: str) -> Path:
    if condition not in DIRECT_RAG_CONDITIONS:
        raise ValueError(f"Unknown Direct RAG condition: {condition}")

    cfg = config.get("direct_rag", {})
    outputs = cfg.get("outputs", {})
    evidence_dir = outputs.get("evidence_dir")

    if not evidence_dir:
        raise KeyError(
            "Missing config entry direct_rag.outputs.evidence_dir. "
            "This must point to the frozen rag_clean/rag_num100 evidence directory."
        )

    return Path(evidence_dir) / f"{condition}.jsonl"


def _load_direct_rag_lookup(
    path: Path,
    condition: str,
) -> dict[tuple[str, int], dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Direct RAG evidence file does not exist for '{condition}':\n{path}"
        )

    lookup: dict[tuple[str, int], dict] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            row = json.loads(line)
            required = {
                "dataset",
                "q_number",
                "condition",
                "evidence_text",
                "evidence_sha256",
            }
            missing = required - set(row)
            if missing:
                raise RuntimeError(
                    f"{condition} evidence line {line_number} is missing "
                    f"required field(s): {sorted(missing)}"
                )

            if str(row["condition"]) != condition:
                raise RuntimeError(
                    f"{condition} evidence line {line_number} has "
                    f"condition={row['condition']!r}"
                )

            dataset = str(row["dataset"])
            q_number = int(row["q_number"])
            evidence = str(row["evidence_text"])

            if not evidence.strip():
                raise RuntimeError(
                    f"Empty Direct RAG evidence for {dataset} q{q_number}"
                )

            expected_hash = _sha256_text(evidence)
            stored_hash = str(row["evidence_sha256"]).strip().lower()
            if stored_hash != expected_hash:
                raise RuntimeError(
                    "Direct RAG evidence SHA-256 mismatch for "
                    f"{condition} {dataset} q{q_number}: "
                    f"stored={stored_hash}, expected={expected_hash}"
                )

            key = (dataset, q_number)
            if key in lookup:
                raise RuntimeError(
                    f"Duplicate Direct RAG evidence record: "
                    f"{condition} {dataset} q{q_number}"
                )

            lookup[key] = row

    if len(lookup) != 200:
        raise RuntimeError(
            f"Expected 200 unique Direct RAG evidence records for "
            f"'{condition}', got {len(lookup)}.\nFile: {path}"
        )

    return lookup


def _prepare_sources(config: dict, conditions: list[str]) -> dict:
    sources = {
        "report_files": {},
        "report_lookups": {},
        "style_file": None,
        "style_lookup": None,
        "csv_direct_file": None,
        "csv_direct_lookup": None,
        "csv_direct_columns": {},
        "numeric_file": None,
        "numeric_lookup": None,
        "direct_rag_files": {},
        "direct_rag_lookups": {},
    }

    if any(c in STYLE_CONDITIONS for c in conditions):
        path, lookup = _load_style_lookup(config)
        sources["style_file"] = path
        sources["style_lookup"] = lookup

    if any(c in {"ev_clean", "ev_conflict"} for c in conditions):
        path, lookup, columns = _load_csv_direct_lookup(config)
        sources["csv_direct_file"] = path
        sources["csv_direct_lookup"] = lookup
        sources["csv_direct_columns"] = columns

    if "ev_numeric" in conditions:
        path = _discover_ev_numeric_file(config)
        sources["numeric_file"] = path
        sources["numeric_lookup"] = _load_ev_numeric_lookup(path)

    for condition in conditions:
        if condition in DIRECT_RAG_CONDITIONS:
            path = _direct_rag_evidence_path(config, condition)
            sources["direct_rag_files"][condition] = path
            sources["direct_rag_lookups"][condition] = _load_direct_rag_lookup(
                path,
                condition,
            )

    for condition in conditions:
        family = _condition_family(condition)
        if family == "report":
            path = _discover_report_file(config, condition)
            sources["report_files"][condition] = path
            sources["report_lookups"][condition] = _load_lookup(path)

    return sources

def _source_for_question(condition: str, question: dict, sources: dict) -> dict:
    dataset, q = _question_key(question)
    family = _condition_family(condition)

    if family == "closedbook":
        return {"evidence_mode": "closedbook", "evidence": "", "source_status": "available", "meta": {}}

    if family == "style":
        style = STYLE_CONDITIONS[condition]
        row = sources["style_lookup"].get((dataset, q, style))
        evidence = _direct_text(row)
        return {
            "evidence_mode": "direct_evidence",
            "evidence": evidence,
            "source_status": "available" if evidence else "unavailable",
            "meta": {
                "evidence_style": style,
                "style_qc_status": None if row is None else row.get("status"),
                "style_numbers_check": None if row is None else row.get("numbers_check"),
                "style_content_check": None if row is None else row.get("content_check"),
            },
        }

    if family == "direct":
        if condition in {"ev_clean", "ev_conflict"}:
            row = (sources["csv_direct_lookup"] or {}).get((dataset, q))
            evidence = "" if row is None else str(row.get(condition, "") or "").strip()
            return {
                "evidence_mode": "direct_evidence",
                "evidence": evidence,
                "source_status": "available" if evidence else "unavailable",
                "meta": {
                    "direct_evidence_source": "benchmark_csv",
                    "direct_evidence_column": sources["csv_direct_columns"].get(condition),
                },
            }

        if condition == "ev_numeric":

            row = (
                sources[
                    "numeric_lookup"
                ]
                .get(
                    (
                        dataset,
                        q,
                    )
                )
            )

            if row is None:
                return {
                    "evidence_mode":
                        "direct_evidence",

                    "evidence":
                        "",

                    "source_status":
                        "unavailable",

                    "meta": {
                        "direct_evidence_source":
                            "ev_numeric_json",
                    },
                }

            evidence = str(
                row["text"]
            ).strip()

            return {
                "evidence_mode": "direct_evidence",
                "evidence": evidence,
                "source_status": "available",
                "meta": {
                    "direct_evidence_source": "ev_numeric_json",
                    "text_sha256": row.get("text_sha256"),
                    "n_values_changed": row.get("n_values_changed"),
                },
            }

        raise ValueError(f"Unsupported direct-evidence condition: {condition}")

    if family == "direct_rag":
        row = (sources["direct_rag_lookups"].get(condition) or {}).get(
            (dataset, q)
        )

        if row is None:
            return {
                "evidence_mode": "direct_rag",
                "evidence": "",
                "source_status": "unavailable",
                "meta": {
                    "direct_rag_condition": condition,
                },
            }

        evidence = str(row.get("evidence_text", ""))
        stored_hash = str(row.get("evidence_sha256", "") or "").lower()
        expected_hash = _sha256_text(evidence)

        if stored_hash != expected_hash:
            raise RuntimeError(
                "Direct RAG evidence hash changed after source loading for "
                f"{condition} {dataset} q{q}"
            )

        return {
            "evidence_mode": "direct_rag",
            "evidence": evidence,
            "source_status": "available",
            "meta": {
                "direct_rag_condition": condition,
                "evidence_sha256": stored_hash,
                "evidence_chars": row.get("evidence_chars"),
                "full_evidence_chars": row.get("full_evidence_chars"),
                "truncated": row.get("truncated"),
                "truncated_chars": row.get("truncated_chars"),
                "query_sha256": row.get("query_sha256"),
                "evidence_cache_key": row.get("evidence_cache_key"),
            },
        }

    row = _lookup_record(sources["report_lookups"][condition], dataset, q)
    failed = _report_is_failed(row)

    if failed:
        return {
            "evidence_mode": "closedbook_fallback",
            "evidence": "",
            "source_status": "available",
            "meta": {
                "report_status": None if row is None else row.get(
                    "report_status",
                    row.get("status"),
                ),
                "report_chars": 0,
            },
        }

    evidence = _report_text(row)
    return {
        "evidence_mode": "report",
        "evidence": evidence,
        "source_status": "available",
        "meta": {
            "report_status": row.get("report_status", row.get("status")),
            "report_chars": len(evidence),
        },
    }


def _answer_output_dir(config: dict, model_name: str) -> Path:
    return _outputs_root(config) / "answers" / _model_slug(model_name)


def _condition_output_path(
    config: dict,
    model_name: str,
    condition: str,
) -> Path:
    return _answer_output_dir(config, model_name) / f"{condition}.json"


def _load_condition_level_existing(
    config: dict,
    model_name: str,
    conditions: list[str],
) -> dict:
    existing = {}
    for condition in conditions:
        path = _condition_output_path(config, model_name, condition)
        if not path.exists():
            continue
        for row in _normalize_json(_load_json(path)):
            if not {"dataset", "q_number", "condition"}.issubset(row):
                continue
            if _requested_model(row) != model_name:
                continue
            row = _canonical_answer_record(row)
            if str(row.get("condition")) != condition:
                continue
            existing[_answer_key(row)] = row
    return existing


def _save_condition_records(
    config: dict,
    model_name: str,
    condition: str,
    existing: dict,
    selected: list[dict],
) -> Path:
    selected_ids = {
        (str(q["dataset"]), int(q["q_number"]))
        for q in selected
    }
    records = [
        row
        for key, row in existing.items()
        if key[2] == condition
        and key[3] == model_name
        and (key[0], key[1]) in selected_ids
    ]
    records.sort(key=lambda x: (str(x["dataset"]), int(x["q_number"])))

    path = _condition_output_path(config, model_name, condition)
    _save_json(path, records)
    return path


def _canonical_answer_record(row: dict) -> dict:
    old_status = str(row.get("answer_status", row.get("parse_status", ""))).lower()
    if old_status in {"parsed", "valid"}:
        answer_status = "parsed"
    elif old_status in {"unparsed", "malformed"}:
        answer_status = "unparsed"
    else:
        answer_status = "error"

    condition = str(row.get("condition", ""))
    old_mode = str(row.get("evidence_mode", ""))
    evidence_sha256 = str(row.get("evidence_sha256", "") or "")
    if condition == "closedbook" or old_mode == "closedbook_fallback":
        evidence_sha256 = ""

    return {
        "dataset": str(row.get("dataset", "")),
        "q_number": int(row.get("q_number")),
        "condition": condition,
        "model_requested": _requested_model(row),
        "model_served": str(row.get("model_served", "") or ""),
        "evidence_sha256": evidence_sha256,
        "prompt_template_sha256": str(row.get("prompt_template_sha256", "") or ""),
        "response_text": str(row.get("response_text", row.get("raw_response", "")) or ""),
        "reasoning_text": str(row.get("reasoning_text", "") or ""),
        "finish_reason": str(row.get("finish_reason", "") or ""),
        "prompt_tokens": row.get("prompt_tokens"),
        "completion_tokens": row.get("completion_tokens"),
        "answer_letter": row.get("answer_letter", row.get("final_answer")),
        "answer_status": answer_status,
        "confidence": row.get("confidence"),
        "attempts": int(row.get("attempts", 0) or 0),
    }


def _base_record(question: dict, condition: str, model_name: str) -> dict:
    return {
        "dataset": str(question["dataset"]),
        "q_number": int(question["q_number"]),
        "condition": condition,
        "model_requested": model_name,
    }


def _source_unavailable(question: dict, condition: str, model_name: str, source: dict) -> dict:
    return {
        **_base_record(question, condition, model_name),
        "model_served": "",
        "evidence_sha256": "",
        "prompt_template_sha256": "",
        "response_text": "",
        "reasoning_text": "",
        "finish_reason": "",
        "prompt_tokens": None,
        "completion_tokens": None,
        "answer_letter": None,
        "answer_status": "error",
        "confidence": None,
        "attempts": 0,
    }


async def _answer_one(
    config: dict,
    model_name: str,
    question: dict,
    condition: str,
    source: dict,
) -> dict:
    if source["source_status"] != "available":
        return _source_unavailable(question, condition, model_name, source)

    question_text = _extract_question_text(question)
    options_text, labels = _extract_options(question)
    option_range = _option_range(labels)
    evidence_mode = source["evidence_mode"]

    if evidence_mode in {"closedbook", "closedbook_fallback"}:
        prompt = CLOSEDBOOK_PROMPT.format(
            question=question_text,
            options=options_text,
            option_range=option_range,
        )
        evidence_sha256 = ""
        prompt_template = CLOSEDBOOK_PROMPT
    else:
        evidence = str(source.get("evidence", ""))
        if not evidence.strip():
            return _source_unavailable(question, condition, model_name, source)
        prompt = ANSWER_PROMPT.format(
            evidence=evidence,
            question=question_text,
            options=options_text,
            option_range=option_range,
        )
        evidence_sha256 = _sha256_text(evidence)
        prompt_template = ANSWER_PROMPT

    prompt_template_sha256 = _sha256_text(prompt_template)
    max_attempts = int(config["answering"]["max_attempts"])

    for attempt in range(1, max_attempts + 1):
        try:
            model = _build_model(config, model_name)
            response = await model.ainvoke([{"role": "user", "content": prompt}])
            response_text = _response_text(response)
            parsed = _parse_answer(response_text, labels)
            meta = _response_meta(response)

            return {
                **_base_record(question, condition, model_name),
                "model_served": str(meta.get("model_served", "") or ""),
                "evidence_sha256": evidence_sha256,
                "prompt_template_sha256": prompt_template_sha256,
                "response_text": response_text,
                "reasoning_text": _reasoning_text(response),
                "finish_reason": str(meta.get("finish_reason", "") or ""),
                "prompt_tokens": meta.get("prompt_tokens"),
                "completion_tokens": meta.get("completion_tokens"),
                "answer_letter": parsed["final_answer"],
                "answer_status": "parsed" if parsed["parse_status"] == "valid" else "unparsed",
                "confidence": parsed["confidence"],
                "attempts": attempt,
            }

        except Exception:
            if attempt < max_attempts:
                await asyncio.sleep(2)

    return {
        **_base_record(question, condition, model_name),
        "model_served": "",
        "evidence_sha256": evidence_sha256,
        "prompt_template_sha256": prompt_template_sha256,
        "response_text": "",
        "reasoning_text": "",
        "finish_reason": "",
        "prompt_tokens": None,
        "completion_tokens": None,
        "answer_letter": None,
        "answer_status": "error",
        "confidence": None,
        "attempts": max_attempts,
    }


async def run_answering(
    config: dict,
    model_name: str,
    conditions: Iterable[str],
) -> Path:
    model_name = resolve_answer_model(config, model_name)
    conditions = list(dict.fromkeys(str(x).strip() for x in conditions if str(x).strip()))

    unknown = [x for x in conditions if x not in ALL_CONDITIONS]
    if unknown:
        raise ValueError(
            "Unknown condition(s): " + ", ".join(unknown)
            + "\nAllowed:\n  " + "\n  ".join(ALL_CONDITIONS)
        )

    questions = _load_json(Path(config["benchmark"]["runtime_json"]))
    if not isinstance(questions, list):
        raise RuntimeError("Benchmark runtime JSON must contain a list.")
    if len(questions) != 200:
        raise RuntimeError(f"Expected 200 benchmark questions, got {len(questions)}")

    selected = questions

    sources = _prepare_sources(config, conditions)

    output_dir = _answer_output_dir(config, model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_concurrency = int(
        config.get("answering", {}).get("max_concurrency", 2)
    )
    if max_concurrency != 2:
        raise RuntimeError(
            f"Answering requires max_concurrency=2; got {max_concurrency}"
        )
    semaphore = asyncio.Semaphore(max_concurrency)
    save_lock = asyncio.Lock()

    existing = _load_condition_level_existing(
        config=config,
        model_name=model_name,
        conditions=conditions,
    )

    slug = _model_slug(model_name)
    run_tag = "full"
    condition_tag = "__".join(conditions)
    if len(condition_tag) > 120:
        condition_tag = hashlib.sha256(condition_tag.encode("utf-8")).hexdigest()[:16]
    settings_path = (
        _outputs_root(config)
        / "settings"
        / f"answers_{slug}__{run_tag}__{condition_tag}.json"
    )


    fallback_keys = set()

    work_items = []
    counter = 0
    for question in selected:
        for condition in conditions:
            counter += 1
            work_items.append((counter, question, condition))

    async def worker(counter: int, question: dict, condition: str):
        dataset, q_number = _question_key(question)
        key = (dataset, q_number, condition, model_name)

        async with save_lock:
            old = existing.get(key)

        source = _source_for_question(condition, question, sources)
        evidence_mode = source["evidence_mode"]

        if evidence_mode == "closedbook_fallback":
            async with save_lock:
                fallback_keys.add((dataset, q_number, condition, model_name))

        if evidence_mode in {"closedbook", "closedbook_fallback"}:
            current_evidence_sha256 = ""
            prompt_template = CLOSEDBOOK_PROMPT
        else:
            current_evidence_sha256 = _sha256_text(
                str(source.get("evidence", ""))
            )
            prompt_template = ANSWER_PROMPT

        current_prompt_template_sha256 = _sha256_text(prompt_template)
        request_signature = _sha256_text(json.dumps(dict(question=question, model=model_name,
            settings=config["answering"], evidence=current_evidence_sha256,
            prompt=current_prompt_template_sha256,
            parser=hashlib.sha256(Path(__file__).with_name("parsing.py").read_bytes()).hexdigest()),
            sort_keys=True, ensure_ascii=False))

        if (
            old is not None
            and old.get("answer_status") in {"parsed", "unparsed"}
            and old.get("request_signature", request_signature) == request_signature
            and old.get("evidence_sha256") == current_evidence_sha256
            and old.get("prompt_template_sha256")
            == current_prompt_template_sha256
        ):
            return


        async with semaphore:


            result = await _answer_one(
                config=config,
                model_name=model_name,
                question=question,
                condition=condition,
                source=source,
            )

        async with save_lock:
            result["request_signature"] = request_signature
            existing[key] = result
            _save_condition_records(
                config=config,
                model_name=model_name,
                condition=condition,
                existing=existing,
                selected=selected,
            )


    await asyncio.gather(
        *[
            worker(counter, question, condition)
            for counter, question, condition in work_items
        ]
    )

    output_paths = {}
    run_records = []
    for condition in conditions:
        path = _save_condition_records(
            config=config,
            model_name=model_name,
            condition=condition,
            existing=existing,
            selected=selected,
        )
        output_paths[condition] = str(path)

        for question in selected:
            key = (
                str(question["dataset"]),
                int(question["q_number"]),
                condition,
                model_name,
            )
            row = existing.get(key)
            if row is not None:
                run_records.append(row)

    settings = {
        "run_type": "condition_level_answering",
        "git_commit": get_git_commit(config["project"]["repo_root"]),
        "config": sanitize_config(config),
        "model_requested": model_name,
        "model_slug": slug,
        "conditions": conditions,
        "n_questions": len(selected),
        "expected_records": len(selected) * len(conditions),
        "max_concurrency": max_concurrency,
        "answer_prompt": ANSWER_PROMPT,
        "closedbook_prompt": CLOSEDBOOK_PROMPT,
        "answering_rules": {
            "unparsed_retry": False,
            "max_concurrency": max_concurrency,
            "json_writes_serialized_with_lock": True,
            "api_failure_max_attempts": int(config["answering"]["max_attempts"]),
            "any_failed_report_condition": "closedbook_fallback",
            "other_missing_required_source": "error",
            "qwen35_9b_enable_thinking": False,
            "qwen35_122b_enable_thinking": False,
            "report_models_source_ablation_conditions_enabled": list(REPORT_MODELS_SOURCE_ABLATION_REPORTS),
            "direct_rag_conditions_enabled": list(DIRECT_RAG_CONDITIONS),
            "resume_requires_evidence_sha256_match": True,
            "resume_requires_prompt_template_sha256_match": True,
        },
        "source_files": {
            "style": None if sources["style_file"] is None else str(sources["style_file"]),
            "direct": {
                "ev_clean": None if sources["csv_direct_file"] is None else str(sources["csv_direct_file"]),
                "ev_conflict": None if sources["csv_direct_file"] is None else str(sources["csv_direct_file"]),
                "ev_numeric": None if sources["numeric_file"] is None else str(sources["numeric_file"]),
            },
            "direct_rag": {
                c: str(p) for c, p in sources["direct_rag_files"].items()
            },
            "reports": {c: str(p) for c, p in sources["report_files"].items()},
        },
        "answer_files": output_paths,
    }
    _save_json(settings_path, settings)


    return output_dir


def main_answers(config_path, model_name, conditions):
    return asyncio.run(run_answering(load_config(config_path), model_name=model_name, conditions=conditions))
