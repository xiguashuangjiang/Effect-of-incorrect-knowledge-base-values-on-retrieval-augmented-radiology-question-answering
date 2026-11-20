"""
src/recheck_style_content_2.py
Created on November 10, 2026

@author: Yangtingting Wang
https://github.com/xiguashuangjiang
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx
from langchain_openai import ChatOpenAI

from src.config import load_config

CHECK_PROMPT_2 = """Original text:
{evidence_clean}

Rewritten text:
{rewrite}

Apart from the tone and the degree of certainty, does the rewritten text contain the same medical claims as the original text, with nothing added and nothing removed? Answer with one word: YES, NO, or PARTIALLY."""
STYLES = ("forum", "overconfident", "hedged", "review")
ALLOWED = {"YES", "NO", "PARTIALLY"}
LABEL_RE = re.compile(r"\b(YES|NO|PARTIALLY)\b", flags=re.I)


def _save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
        fp.flush()
        os.fsync(fp.fileno())
    temporary.replace(path)


def _clean_evidence_map(source_path: Path) -> dict[tuple[str, int], str]:
    with source_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        if not {"dataset", "q_number", "evidence_clean"}.issubset(reader.fieldnames or []):
            raise ValueError("Benchmark CSV missing dataset/q_number/evidence_clean")
        result = {}
        for row in reader:
            key = (row["dataset"].strip(), int(row["q_number"]))
            if key in result:
                raise ValueError(f"Duplicate benchmark key: {key}")
            result[key] = row["evidence_clean"]
    if len(result) != 200:
        raise ValueError(f"Expected 200 benchmark questions, got {len(result)}")
    return result


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _verify_unchanged(old: list[dict], new: list[dict]) -> None:
    if len(old) != len(new):
        raise RuntimeError("Record count changed")
    for before, after in zip(old, new):
        for key, value in before.items():
            if key != "content_check_2" and after.get(key) != value:
                raise RuntimeError(f"Original field {key} changed for {before.get('dataset')} {before.get('q_number')}")


def _git_hash(root: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _response_label(response) -> tuple[str, dict]:
    content = response.content
    if not isinstance(content, str):
        content = str(content)
    label = content.strip().strip(". \n\t").upper()
    if label not in ALLOWED:
        match = LABEL_RE.fullmatch(content.strip())
        label = match.group(1).upper() if match else "UNPARSED"
    meta = response.response_metadata or {}
    usage = response.usage_metadata or {}
    return label, {
        "model_served": meta.get("model_name") or meta.get("model") or "",
        "finish_reason": meta.get("finish_reason"),
        "prompt_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens"),
    }


def run_content_check_2(config: dict) -> dict:
    part_dir = Path(config["outputs"]["corruption_direct_evidence"])
    repo = Path(config["project"]["repo_root"]).resolve()
    output_path = (part_dir / "texts" / "ev_styles.json").resolve()
    try:
        output_path.relative_to(repo)
    except ValueError:
        pass
    else:
        raise ValueError("ev_styles.json is located inside the Git repository; move outputs outside")
    if not output_path.exists():
        raise FileNotFoundError(output_path)
    original_bytes = output_path.read_bytes()
    records = json.loads(original_bytes)
    if not isinstance(records, list) or len(records) != 800:
        raise ValueError(f"Expected exactly 800 style records, found {len(records) if isinstance(records,list) else 'non-list'}")
    evidence_map = _clean_evidence_map(Path(config["benchmark"]["source_csv"]))
    seen = set()
    for rec in records:
        key = (str(rec["dataset"]), int(rec["q_number"]), str(rec["style"]))
        if key in seen or key[:2] not in evidence_map or key[2] not in STYLES:
            raise ValueError(f"Invalid or duplicate style record: {key}")
        seen.add(key)
        if _sha256(rec["text"]) != rec["text_sha256"]:
            raise ValueError(f"Text SHA mismatch: {key}")
    if len(seen) != 800:
        raise ValueError("Incomplete style conditions")

    check_cfg = config["content_check_2"]
    before = json.loads(original_bytes)

    requested_model = config["models"]["checker_model"]
    model_api = requested_model.split(":", 1)[-1] if requested_model.startswith("custom:") else requested_model
    max_tokens = int(check_cfg["max_tokens"])
    model = ChatOpenAI(
        base_url=config["api"]["custom_api_url"].rstrip("/"),
        api_key=os.getenv(config["api"]["api_key_env"], "sk-dummy-key"),
        model=model_api,
        temperature=float(check_cfg["temperature"]),
        max_tokens=max_tokens,
        timeout=float(check_cfg["timeout_seconds"]),
        max_retries=0,
        streaming=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        http_client=httpx.Client(verify=False),
    )

    settings_path = part_dir / "settings" / "ev_styles_content_check_2.json"
    log_path = part_dir / "logs" / "style_content_check_2_calls.json"
    logs = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    completed = 0
    for i, rec in enumerate(records, 1):
        if rec.get("content_check_2") in ALLOWED:
            completed += 1
            continue
        key = (str(rec["dataset"]), int(rec["q_number"]))
        prompt = CHECK_PROMPT_2.format(evidence_clean=evidence_map[key], rewrite=rec["text"])
        result = "UNPARSED"
        metadata = {}
        error = None
        for attempt in range(1, int(check_cfg["max_attempts"]) + 1):
            try:
                response = model.invoke([{"role": "user", "content": prompt}])
                result, metadata = _response_label(response)
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        logs.append({"dataset": key[0], "q_number": key[1], "style": rec["style"], "response": result,
                     "error": error, "checker_model_requested": requested_model, **metadata})
        if result in ALLOWED:
            rec["content_check_2"] = result
            completed += 1
        _verify_unchanged(before, records)
        _save(output_path, records)
        _save(log_path, logs)

    counts = {style: Counter(r.get("content_check_2", "MISSING") for r in records if r["style"] == style)
              for style in STYLES}
    settings = {
        "run_type": "evidence_style_content_check_2", "git_commit": _git_hash(repo),
        "config": {k: v for k, v in config.items() if k != "api"},
        "api_config": {"api_url_env": config["api"]["api_url_env"], "api_key_env": config["api"]["api_key_env"]},
        "checker_model_requested": requested_model,
        "checker_models_served": sorted({x["model_served"] for x in logs if x.get("model_served")}),
        "checker_prompt": CHECK_PROMPT_2, "checker_temperature": float(check_cfg["temperature"]), "checker_max_tokens": max_tokens,
        "checker_chat_template_kwargs": {"enable_thinking": False}, "timeout_seconds": float(check_cfg["timeout_seconds"]), "max_attempts": int(check_cfg["max_attempts"]),
        "output_file": str(output_path), "input_benchmark": str(config["benchmark"]["source_csv"]),
        "records": 800, "completed": completed,
        "per_style": {k: dict(v) for k, v in counts.items()},
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    _save(settings_path, settings)
    return settings


def main_content_check(config_path):
    return run_content_check_2(load_config(config_path))
