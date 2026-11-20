"""
src/report_generation.py
Created on October 22, 2026

@author: Yangtingting Wang
https://github.com/xiguashuangjiang
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.config import load_config
from src.metadata import get_git_commit, sanitize_config


FORMAL_REPORT_MODEL = "nvidia/Qwen3.8-Flash-Next-NVFP4"

BASE_REPORT_CONDITIONS = {
    "clean": {"corpus": "clean", "noretrieval": False, "volume": None},
    "num010": {"corpus": "num010", "noretrieval": False, "volume": None},
    "num025": {"corpus": "num025", "noretrieval": False, "volume": None},
    "num050": {"corpus": "num050", "noretrieval": False, "volume": None},
    "num100": {"corpus": "num100", "noretrieval": False, "volume": None},
    "cov010": {"corpus": "cov010", "noretrieval": False, "volume": None},
    "cov025": {"corpus": "cov025", "noretrieval": False, "volume": None},
    "cov050": {"corpus": "cov050", "noretrieval": False, "volume": None},
    "cov100": {"corpus": "cov100", "noretrieval": False, "volume": None},
    "clean_r2": {"corpus": "clean", "noretrieval": False, "volume": None},
    "clean_r3": {"corpus": "clean", "noretrieval": False, "volume": None},
    "noretrieval": {"corpus": "clean", "noretrieval": True, "volume": None},
    "top1_100": {"corpus": "top1_100", "noretrieval": False, "volume": None},
    "topk01_pad00": {"corpus": "clean", "noretrieval": False, "volume": (1, 0.00)},
    "topk03_pad00": {"corpus": "clean", "noretrieval": False, "volume": (3, 0.00)},
    "topk05_pad00": {"corpus": "clean", "noretrieval": False, "volume": (5, 0.00)},
    "topk10_pad00": {"corpus": "clean", "noretrieval": False, "volume": (10, 0.00)},
    "topk20_pad00": {"corpus": "clean", "noretrieval": False, "volume": (20, 0.00)},
    "topk05_pad50": {"corpus": "clean", "noretrieval": False, "volume": (5, 0.50)},
    "topk05_pad75": {"corpus": "clean", "noretrieval": False, "volume": (5, 0.75)},
    "topk05_pad90": {"corpus": "clean", "noretrieval": False, "volume": (5, 0.90)},
    "topk05_pad95": {"corpus": "clean", "noretrieval": False, "volume": (5, 0.95)},
}


REPORT_MODELS_SOURCE_ABLATION_REPORT_MODELS = {
    "dsv4": "MJPansa/DeepSeek-V4-Flash-0731-NVFP4",
    "q35": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
}

REPORT_MODELS_SOURCE_ABLATION_REPORT_CONDITIONS = {
    f"{prefix}_{kind}": {
        "corpus": corpus,
        "noretrieval": kind == "noretrieval",
        "volume": None,
        "model": model,
    }
    for prefix, model in REPORT_MODELS_SOURCE_ABLATION_REPORT_MODELS.items()
    for kind, corpus in (
        ("clean", "clean"),
        ("clean_r2", "clean"),
        ("num100", "num100"),
        ("cov100", "cov100"),
        ("noretrieval", "clean"),
    )
}

def _reports_root(config: dict) -> Path:
    return Path(
        config.get("outputs", {}).get(
            "formal_reports",
            Path(config["paths"]["output_root"]) / "reports",
        )
    )


def _settings_root(config: dict) -> Path:
    return Path(
        config.get("outputs", {}).get(
            "formal_report_settings",
            Path(config["paths"]["output_root"]) / "report_settings",
        )
    )


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _key(record: dict) -> tuple[str, int]:
    return str(record["dataset"]), int(record["q_number"])


def _load_existing(path: Path) -> dict[tuple[str, int], dict]:
    if not path.exists():
        return {}
    data = _load_json(path)
    if not isinstance(data, list):
        raise RuntimeError(f"Expected list in {path}")
    out = {}
    for row in data:
        key = _key(row)
        if key in out:
            raise RuntimeError(f"Duplicate question key in {path}: {key}")
        out[key] = row
    return out


def _resolve_corpus(config: dict, key: str) -> Path:
    if key == "clean":
        return Path(config["corpora"]["clean"])
    if key.startswith("num"):
        return Path(config["corpora"]["numerical"][key])
    if key.startswith("cov"):
        return Path(config["corpora"]["coverage"][key])
    if key == "top1_100":
        return Path(config["paths"]["corpus_root"]) / "source_ablation" / "top1_100"
    raise ValueError(f"Unknown corpus key: {key}")


def _corpus_fingerprint(corpus: Path) -> tuple[str, int]:
    rows = []
    files = sorted(corpus.glob("*.txt"))
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{path.name}\t{digest}")
    joined = "\n".join(rows)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest(), len(files)


def _runtime_model_name(name: str) -> str:
    return name if name.startswith("custom:") else f"custom:{name}"


def _formal_run_config(config: dict, report_model: str = FORMAL_REPORT_MODEL) -> dict:
    run_config = deepcopy(config)
    run_config["models"]["report_model"] = _runtime_model_name(report_model)
    if report_model == FORMAL_REPORT_MODEL:
        run_config.setdefault("report_generation", {})["reasoning_effort"] = "none"
    return run_config


def _load_pipeline(config: dict):
    repo_root = Path(config["project"]["repo_root"])
    legacy_dir = repo_root / "rar_pipeline"
    if not legacy_dir.exists():
        raise FileNotFoundError(f"RaR pipeline directory not found: {legacy_dir}")
    legacy_dir_str = str(legacy_dir)
    if legacy_dir_str not in sys.path:
        sys.path.insert(0, legacy_dir_str)

    os.environ["RESEARCHER_MODEL"] = config["models"]["report_model"]
    os.environ["CUSTOM_API_URL"] = config["api"]["custom_api_url"]
    api_key_env = config["api"]["api_key_env"]
    if api_key_env not in os.environ:
        os.environ[api_key_env] = ""

    import runner
    import local_search
    import agentic_workflow
    return runner, local_search, agentic_workflow


def _install_noretrieval(local_search, agentic_workflow) -> None:
    from src.noretrieval import _install_noretrieval_behavior
    _install_noretrieval_behavior(local_search, agentic_workflow)


def _input_signature(
    *,
    condition: str,
    question: dict,
    corpus_fingerprint: str,
    config: dict,
    spec: dict,
    report_model: str = FORMAL_REPORT_MODEL,
) -> str:
    payload = {
        "condition": condition,
        "dataset": str(question["dataset"]),
        "q_number": int(question["q_number"]),
        "question": question.get("summary", ""),
        "options": question.get("options", {}),
        "corpus_fingerprint": corpus_fingerprint,
        "report_model": report_model,
        "reasoning_effort": "none" if report_model == FORMAL_REPORT_MODEL else None,
        "temperature": config["report_generation"]["temperature"],
        "max_tokens": config["report_generation"]["max_tokens"],
        "max_search_depth": config["report_generation"]["max_search_depth"],
        "noretrieval": bool(spec["noretrieval"]),
        "volume": spec["volume"],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _has_length_finish_reason(calls: list[dict]) -> bool:
    return any(
        str(call.get("finish_reason") or "").strip().lower() == "length"
        for call in (calls or [])
    )


async def _generate_one(
    *,
    config: dict,
    question: dict,
    runner,
    local_search,
) -> dict:
    question_text = question.get("summary", "")
    options_text = "\n".join(
        f"{label}: {text}"
        for label, text in question.get("options", {}).items()
    )
    max_attempts = int(config["report_generation"]["max_attempts"])
    timeout_seconds = int(config["report_generation"]["timeout_seconds"])
    last_error = None
    last_log = []

    for attempt in range(1, max_attempts + 1):
        local_search.clear_retrieval_log()
        try:
            graph_config = dict(runner.config)
            graph_config["configurable"] = dict(graph_config.get("configurable", {}))
            for setting in ["temperature", "max_tokens", "max_search_depth"]:
                graph_config["configurable"][setting] = config["report_generation"][setting]
            graph_config["recursion_limit"] = int(
                config["report_generation"]["recursion_limit"]
            )
            generation = await asyncio.wait_for(
                runner.run_agenticRAG_detailed(
                    question_text,
                    options_text,
                    graph_config,
                ),
                timeout=timeout_seconds,
            )
            report = generation["report"]
            calls = generation.get("report_model_calls", [])
            last_log = local_search.get_retrieval_log()

            if not report or not str(report).strip() or str(report).strip().lower().startswith("[error]"):
                raise RuntimeError("No valid final report was produced.")

            if _has_length_finish_reason(calls):
                raise RuntimeError(
                    "Report generation was truncated: at least one report-model "
                    "call ended with finish_reason='length'."
                )

            return {
                "status": "complete",
                "attempts": attempt,
                "report": report,
                "retrieval_log": last_log,
                "report_model_calls": calls,
                "error": None,
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            last_log = local_search.get_retrieval_log()

    return {
        "status": "failed",
        "attempts": max_attempts,
        "report": None,
        "retrieval_log": last_log,
        "report_model_calls": [],
        "error": last_error,
    }


async def run_report_condition(
    config: dict,
    *,
    condition: str,
) -> Path:
    is_report_models_source_ablation_model = condition in REPORT_MODELS_SOURCE_ABLATION_REPORT_CONDITIONS
    if condition not in BASE_REPORT_CONDITIONS and not is_report_models_source_ablation_model:
        raise ValueError(f"Unsupported report condition: {condition}")

    spec = (REPORT_MODELS_SOURCE_ABLATION_REPORT_CONDITIONS[condition] if is_report_models_source_ablation_model
            else BASE_REPORT_CONDITIONS[condition])
    report_model = spec.get("model", FORMAL_REPORT_MODEL)
    if is_report_models_source_ablation_model and report_model not in config.get("models", {}).get("additional_report_generators", []):
        raise RuntimeError(f"Unconfigured report variants report model: {report_model}")
    corpus = _resolve_corpus(config, spec["corpus"])
    if not corpus.exists():
        raise FileNotFoundError(f"Corpus does not exist for {condition}: {corpus}")

    questions = _load_json(config["benchmark"]["runtime_json"])
    if not isinstance(questions, list) or len(questions) != 200:
        raise RuntimeError("Expected exactly 200 benchmark questions.")
    selected = questions

    run_config = _formal_run_config(config, report_model=report_model)
    runner, local_search, agentic_workflow = _load_pipeline(run_config)

    local_search.set_corpus_dir(str(corpus))
    volume = spec["volume"]
    if volume is not None:
        top_k, padding_ratio = volume
        local_search.configure_volume_mode(
            enabled=True,
            top_k=top_k,
            padding_ratio=padding_ratio,
            seed=int(config["retrieval"]["volume"]["padding_seed"]),
        )
    else:
        local_search.configure_volume_mode(enabled=False)

    if spec["noretrieval"]:
        _install_noretrieval(local_search, agentic_workflow)

    corpus_fp, corpus_n = _corpus_fingerprint(corpus)
    package_root = Path(run_config["project"]["repo_root"])
    implementation_files = sorted((package_root / "rar_pipeline").glob("*.py"))
    implementation_files += [package_root / "src" / name for name in ["report_generation.py", "noretrieval.py"]]
    implementation_signature = hashlib.sha256(json.dumps(dict(
        sources={str(path.relative_to(package_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in implementation_files},
        report_settings=run_config["report_generation"], retrieval=run_config["retrieval"],
        api_url=run_config["api"]["custom_api_url"]), sort_keys=True).encode("utf-8")).hexdigest()

    if is_report_models_source_ablation_model:
        report_models_source_ablation_root = Path(config.get("outputs", {}).get(
            "report_models_source_ablation", Path(config["paths"]["output_root"]) / "report_models_source_ablation"))
        reports_root = report_models_source_ablation_root / "reports"
        settings_root = report_models_source_ablation_root / "settings"
        if spec["corpus"] == "cov100" and corpus_n != 8097:
            raise RuntimeError(f"Fixed Coverage100 must contain 8097 txt files, got {corpus_n}")
    else:
        reports_root = _reports_root(config)
        settings_root = _settings_root(config)
    output_path = reports_root / f"{condition}.json"
    settings_path = settings_root / f"{condition}.json"

    if condition == "top1_100" and corpus_n != 16685:
        raise RuntimeError(
            f"Corrected top1_100 must contain 16685 articles, got {corpus_n}. "
            "Do not generate reports from an unverified corpus."
        )

    existing = _load_existing(output_path)
    if is_report_models_source_ablation_model:
        unsigned = [key for key, row in existing.items()
                    if not row.get("input_signature_sha256")]
        if unsigned:
            raise RuntimeError(
                f"{output_path} contains {len(unsigned)} records without an input signature. "
                "Move the file before regenerating; no file was changed."
            )
    concurrency = int(config["report_generation"].get("max_concurrency", 2))
    if concurrency != 2:
        raise RuntimeError(f"Report generation requires max_concurrency=2; got {concurrency}")

    semaphore = asyncio.Semaphore(concurrency)
    save_lock = asyncio.Lock()
    start = time.perf_counter()

    async def worker(index: int, question: dict):
        key = _key(question)
        signature = _input_signature(
            condition=condition,
            question=question,
            corpus_fingerprint=corpus_fp,
            config=run_config,
            spec=spec,
            report_model=report_model,
        )
        old = existing.get(key)

        signature_match = (
            old is not None
            and old.get("report_status") == "complete"
            and str(old.get("report", "")).strip()
            and old.get("input_signature_sha256") == signature
            and old.get("implementation_signature_sha256", implementation_signature) == implementation_signature
            and not _has_length_finish_reason(old.get("report_model_calls", []))
            and (not is_report_models_source_ablation_model or (
                old.get("corpus_fingerprint_sha256") == corpus_fp
                and old.get("report_model_requested") == report_model
            ))
        )

        if signature_match:
            return


        async with semaphore:
            result = await _generate_one(
                config=run_config,
                question=question,
                runner=runner,
                local_search=local_search,
            )
            if is_report_models_source_ablation_model and result["status"] == "complete":
                served = {
                    str(call.get("model_served") or call.get("model") or "")
                    .removeprefix("custom:")
                    for call in result.get("report_model_calls", [])
                    if call.get("model_served") or call.get("model")
                }
                if served and served != {report_model}:
                    result["status"] = "failed"
                    result["error"] = f"Unexpected model served: {sorted(served)}"
                    result["report"] = None

        record = dict(question)
        record.update({
            "condition": condition,
            "corpus_variant": "noretrieval" if spec["noretrieval"] else corpus.name,
            "corpus_fingerprint_sha256": corpus_fp,
            "input_signature_sha256": signature,
            "implementation_signature_sha256": implementation_signature,
            "report_model_requested": report_model,
            "effective_temperature": run_config["report_generation"]["temperature"],
        "effective_max_tokens": run_config["report_generation"]["max_tokens"],
            "report": result["report"],
            "retrieval_log": result["retrieval_log"],
            "report_status": result["status"],
            "attempts": result["attempts"],
            "report_model_calls": result.get("report_model_calls", []),
            "report_error": result.get("error"),
        })

        async with save_lock:
            existing[key] = record
            ordered = [existing[_key(q)] for q in selected if _key(q) in existing]
            _save_json(output_path, ordered)

    await asyncio.gather(
        *[worker(i, q) for i, q in enumerate(selected, start=1)]
    )

    final = _load_existing(output_path)
    all_calls = [
        call
        for row in final.values()
        for call in row.get("report_model_calls", [])
    ]
    served_names = sorted({
        str(call.get("model_served") or call.get("model") or "")
        for call in all_calls
        if call.get("model_served") or call.get("model")
    })
    finish_reasons = [
        str(call.get("finish_reason") or "")
        for call in all_calls
    ]

    settings = {
        "run_type": "formal_report_generation",
        "condition": condition,
        "git_commit": get_git_commit(config["project"]["repo_root"]),
        "config": sanitize_config(config),
        "model_requested": report_model,
        "model_runtime": run_config["models"]["report_model"],
        "model_served_names": served_names,
        "model_request_overrides": ({"enable_thinking": False} if "Qwen3.5-35B-A3B" in report_model else {}),
        "effective_temperature": run_config["report_generation"]["temperature"],
        "effective_max_tokens": run_config["report_generation"]["max_tokens"],
        "corpus": str(corpus),
        "corpus_files": corpus_n,
        "corpus_fingerprint_sha256": corpus_fp,
        "noretrieval": bool(spec["noretrieval"]),
        "volume": spec["volume"],
        "max_concurrency": concurrency,
        "option_sections_sequential": True,
        "n_questions": len(selected),
        "complete": sum(r.get("report_status") == "complete" for r in final.values()),
        "failed": sum(r.get("report_status") == "failed" for r in final.values()),
        "finish_reason_length_calls": sum(x.lower() == "length" for x in finish_reasons),
        "elapsed_seconds": round(time.perf_counter() - start, 3),
    }
    _save_json(settings_path, settings)


    return output_path


def main_reports(config_path, condition):
    for name in ("runner", "agentic_workflow", "local_search", "utils", "configuration", "prompts_radiology"):
        sys.modules.pop(name, None)
    return asyncio.run(run_report_condition(load_config(config_path), condition=condition))
