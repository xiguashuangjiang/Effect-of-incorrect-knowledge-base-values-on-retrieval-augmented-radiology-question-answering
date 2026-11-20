"""
src/evidence_numeric.py
Created on October 13, 2026

@author: Yangtingting Wang
https://github.com/xiguashuangjiang
"""

import csv
import hashlib
import json
import random
import subprocess
from pathlib import Path

from scripts.numeric_patterns import detect_clinical_numbers
from src.config import load_config
from scripts.numerical_perturbation import (
    get_article_seed,
    perturb_text,
)


FORMAL_SCHEMA = [
    "dataset",
    "q_number",
    "text",
    "text_sha256",
    "n_values_changed",
]


def load_benchmark(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sanitize_config(value):
    secret_terms = ("api_key", "apikey", "token", "secret", "password")

    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if any(term in key_text for term in secret_terms):
                out[key] = "***REDACTED***"
            else:
                out[key] = _sanitize_config(item)
        return out

    if isinstance(value, list):
        return [_sanitize_config(x) for x in value]

    return value


def _git_commit(config: dict) -> str | None:
    repo_root = config.get("project", {}).get("repo_root")
    if not repo_root:
        return None

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() or None
    except Exception:
        return None


def build_ev_numeric(config: dict) -> dict:
    benchmark_path = Path(config["benchmark"]["source_csv"])
    rows = load_benchmark(benchmark_path)

    if len(rows) != 200:
        raise ValueError(
            f"Expected 200 benchmark questions, got {len(rows)}"
        )

    seen_ids = set()
    records = []
    all_changes = []

    questions_with_numbers = 0
    questions_without_numbers = 0
    questions_changed = 0

    for row in rows:
        dataset = row["dataset"].strip()
        q_number = int(row["q_number"])
        question_id = (dataset, q_number)

        if question_id in seen_ids:
            raise ValueError(
                f"Duplicate benchmark ID: {question_id}"
            )
        seen_ids.add(question_id)

        evidence_clean = row.get("evidence_clean")
        if evidence_clean is None:
            raise ValueError(
                f"Missing evidence_clean: {dataset} q{q_number}"
            )
        evidence_clean = str(evidence_clean)

        article_id = f"RadSaFE_{dataset}_{q_number}"
        detected = detect_clinical_numbers(evidence_clean)

        if not detected:
            evidence_numeric = evidence_clean
            audit_entries = []
            questions_without_numbers += 1
        else:
            questions_with_numbers += 1
            random.seed(get_article_seed(article_id))
            evidence_numeric, audit_entries = perturb_text(
                evidence_clean,
                article_id=article_id,
            )
            if evidence_numeric != evidence_clean:
                questions_changed += 1

        if not detected and evidence_numeric != evidence_clean:
            raise RuntimeError(
                "Text changed despite no detected numbers: "
                f"{dataset} q{q_number}"
            )

        all_changes.extend(audit_entries)

        records.append(
            {
                "dataset": dataset,
                "q_number": q_number,
                "text": evidence_numeric,
                "text_sha256": sha256_text(evidence_numeric),
                "n_values_changed": len(audit_entries),
            }
        )

    if len(seen_ids) != 200:
        raise RuntimeError(
            f"Expected 200 unique IDs, got {len(seen_ids)}"
        )

    return {
        "records": records,
        "changes": all_changes,
        "summary": {
            "total_questions": len(records),
            "unique_ids": len(seen_ids),
            "questions_with_detected_numbers": questions_with_numbers,
            "questions_without_detected_numbers": questions_without_numbers,
            "questions_changed": questions_changed,
            "total_perturbations": len(all_changes),
            "formal_schema": FORMAL_SCHEMA,
        },
    }


def save_ev_numeric(result: dict, config: dict) -> dict:
    corruption_direct_evidence_root = Path(config["outputs"]["corruption_direct_evidence"])
    texts_dir = corruption_direct_evidence_root / "texts"
    logs_dir = corruption_direct_evidence_root / "logs"
    settings_dir = corruption_direct_evidence_root / "settings"

    for directory in (texts_dir, logs_dir, settings_dir):
        directory.mkdir(parents=True, exist_ok=True)

    text_path = texts_dir / "ev_numeric.json"
    audit_path = logs_dir / "ev_numeric_changes.json"
    summary_path = logs_dir / "ev_numeric_summary.json"
    settings_path = settings_dir / "ev_numeric.json"

    with text_path.open("w", encoding="utf-8") as f:
        json.dump(
            result["records"],
            f,
            ensure_ascii=False,
            indent=2,
        )

    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(
            result["changes"],
            f,
            ensure_ascii=False,
            indent=2,
        )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            result["summary"],
            f,
            ensure_ascii=False,
            indent=2,
        )

    benchmark_path = Path(config["benchmark"]["source_csv"])

    settings = {
        "run_type": "ev_numeric",
        "git_commit": _git_commit(config),
        "config": _sanitize_config(config),
        "method": {
            "type": "deterministic_rule_based_perturbation",
            "source_field": "evidence_clean",
            "detector": "scripts.numeric_patterns.detect_clinical_numbers",
            "perturbation_function": "scripts.numerical_perturbation.perturb_text",
            "seed_function": "scripts.numerical_perturbation.get_article_seed",
            "seed_article_id_format": "RadSaFE_<dataset>_<q_number>",
            "llm_used": False,
        },
        "inputs": {
            "benchmark_path": str(benchmark_path),
            "benchmark_sha256": sha256_file(benchmark_path),
        },
        "outputs": {
            "ev_numeric": {
                "path": str(text_path),
                "sha256": sha256_file(text_path),
                "records": len(result["records"]),
            },
            "audit": {
                "path": str(audit_path),
                "sha256": sha256_file(audit_path),
                "records": len(result["changes"]),
            },
            "summary": {
                "path": str(summary_path),
                "sha256": sha256_file(summary_path),
            },
        },
    }

    settings_canonical = json.dumps(
        settings,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    settings["settings_sha256"] = sha256_text(settings_canonical)

    with settings_path.open("w", encoding="utf-8") as f:
        json.dump(
            settings,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "text": text_path,
        "evidence": text_path,
        "audit": audit_path,
        "summary": summary_path,
        "settings": settings_path,
    }


def main_changed_evidence(config_path):
    config = load_config(config_path)
    result = build_ev_numeric(config)
    return save_ev_numeric(result, config)
