"""
src/source_ablation_corpus.py
Created on October 26, 2026

@author: Yangtingting Wang
https://github.com/xiguashuangjiang
"""

from __future__ import annotations

import hashlib
import shutil
from datetime import datetime
from uuid import uuid4
from collections import Counter
from pathlib import Path

from src.config import load_config
from src.cited_article_pool import (
    _report_models_source_ablation_root,
    _report_text,
    build_corpus_index,
    extract_sources_from_report,
    load_json,
    resolve_article,
    save_json,
)


def _new_clean_report_path(config: dict) -> Path:
    configured = (
        config.get("coverage_rebuild", {})
        .get("source_clean_reports")
    )

    if configured:
        path = Path(configured)
    else:
        formal_root = Path(
            config.get("outputs", {}).get(
                "formal_reports",
                Path(config["paths"]["output_root"]) / "reports",
            )
        )
        path = formal_root / "clean.json"

    if not path.exists():
        raise FileNotFoundError(
            "New formal clean report JSON not found:\n"
            f"{path}\n"
            "Finish the new clean 200 run before rebuilding top1_100."
        )

    return path


def _corpus_root(config: dict) -> Path:
    return Path(config["paths"]["corpus_root"]) / "source_ablation"


def _most_cited_first_tie(article_ids: list[str]) -> str | None:
    if not article_ids:
        return None

    counts = Counter(article_ids)
    highest = max(counts.values())

    for article_id in article_ids:
        if counts[article_id] == highest:
            return article_id

    return None


def build_report_models_source_ablation_top1_corpus(config: dict) -> Path:
    clean = Path(config["corpora"]["clean"])
    report_path = _new_clean_report_path(config)
    records = load_json(report_path)

    if not isinstance(records, list) or len(records) != 200:
        raise RuntimeError(
            f"Expected 200 clean report records, got "
            f"{len(records) if isinstance(records, list) else 'non-list'}"
        )

    keys = [
        (str(record.get("dataset", "")), int(record.get("q_number")))
        for record in records
    ]
    if len(set(keys)) != 200:
        raise RuntimeError(
            "New clean report set does not contain 200 unique "
            "(dataset, q_number) pairs."
        )

    incomplete = [
        key
        for key, record in zip(keys, records)
        if record.get("report_status") != "complete"
    ]
    if incomplete:
        raise RuntimeError(
            f"New clean report set contains {len(incomplete)} incomplete/"
            f"failed reports; first few: {incomplete[:10]}"
        )

    corpus_index, hash_index = build_corpus_index(clean)
    per_question = []
    union_to_remove: set[Path] = set()
    all_fallbacks = []

    for record in records:
        dataset = str(record.get("dataset", ""))
        q_number = int(record.get("q_number"))

        raw_ids = extract_sources_from_report(
            _report_text(record)
        )

        resolved = []
        unresolved = []
        fallbacks = []

        for source_id in raw_ids:
            try:
                path = resolve_article(
                    source_id,
                    corpus_index,
                    hash_index,
                    resolution_log=fallbacks,
                )
            except KeyError:
                unresolved.append(source_id)
                continue

            resolved.append(path.name)

        selected = _most_cited_first_tie(resolved)

        if selected is not None:
            selected_path = resolve_article(selected, corpus_index, hash_index)
            union_to_remove.add(selected_path)

        for correction in fallbacks:
            all_fallbacks.append({"dataset": dataset, "q_number": q_number, **correction})

        per_question.append(
            {
                "dataset": dataset,
                "q_number": q_number,
                "sources_raw_in_order": raw_ids,
                "sources_resolved_in_order": resolved,
                "unresolved_sources": unresolved,
                "hash_mismatch_resolutions": fallbacks,
                "removed_article": selected or "",
            }
        )

    if any(row["unresolved_sources"] for row in per_question):
        raise RuntimeError("Unresolved citations remain; no corpus was changed.")

    report_digest = hashlib.sha256(report_path.read_bytes()).hexdigest()

    corpus_root = _corpus_root(config)
    corpus_root.mkdir(parents=True, exist_ok=True)
    target = corpus_root / "top1_100"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    staging = corpus_root / f"top1_100.staging_{run_id}"
    old_corpus_backup = corpus_root / f"top1_100.previous_{run_id}"
    if staging.exists() or old_corpus_backup.exists():
        raise FileExistsError("Staging/backup path collision; no changes made.")

    shutil.copytree(clean, staging)
    removal_log = []

    for source_path in sorted(union_to_remove):
        rel = source_path.relative_to(clean)
        target_path = staging / rel

        if not target_path.exists():
            raise FileNotFoundError(
                f"Expected copied article missing: {target_path}"
            )

        target_path.unlink()

        removal_log.append(
            {
                "article_id": source_path.name,
                "relative_path": rel.as_posix(),
            }
        )

    clean_files = {p.relative_to(clean).as_posix(): p for p in clean.rglob("*.txt")}
    staged_files = {p.relative_to(staging).as_posix(): p for p in staging.rglob("*.txt")}
    clean_count = len(clean_files)
    target_count = len(staged_files)
    expected = clean_count - len(union_to_remove)

    removed_relpaths = {p.relative_to(clean).as_posix() for p in union_to_remove}
    expected_relpaths = set(clean_files) - removed_relpaths
    if set(staged_files) != expected_relpaths:
        raise RuntimeError("Staged corpus filenames differ from clean-minus-removed set.")
    for rel in sorted(expected_relpaths):
        source = clean_files[rel]
        copied = staged_files[rel]
        if source.stat().st_size != copied.stat().st_size:
            raise RuntimeError(f"Staged article size mismatch: {rel}")
        src_hash = hashlib.sha256(source.read_bytes()).digest()
        copied_hash = hashlib.sha256(copied.read_bytes()).digest()
        if src_hash != copied_hash:
            raise RuntimeError(f"Staged article content mismatch: {rel}")
    if target_count != expected:
        raise RuntimeError(
            f"top1_100 article count mismatch: "
            f"{target_count} != expected {expected}"
        )

    unresolved_records = [
        {
            "dataset": row["dataset"],
            "q_number": row["q_number"],
            "source_id": source_id,
        }
        for row in per_question
        for source_id in row.get("unresolved_sources", [])
    ]

    root = _report_models_source_ablation_root(config)
    log_path = root / "logs" / "top1_removal.json"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    staged_log = log_path.with_name(f"top1_removal.staging_{run_id}.json")
    old_log_backup = log_path.with_name(f"top1_removal.previous_{run_id}.json")
    save_json(
        staged_log,
        {
            "source_clean_reports": str(report_path),
            "source_clean_reports_sha256": report_digest,
            "rebuild_run_id": run_id,
            "clean_article_count": clean_count,
            "questions": len(records),
            "unique_removed_articles": len(union_to_remove),
            "top1_article_count": target_count,
            "unresolved_source_count": len(unresolved_records),
            "unresolved_sources": unresolved_records,
            "hash_mismatch_resolution_count": len(all_fallbacks),
            "hash_mismatch_resolutions": all_fallbacks,
            "removed_articles": removal_log,
            "per_question": per_question,
        },
    )

    corpus_moved = False
    new_corpus_installed = False
    log_moved = False
    try:
        if target.exists():
            target.rename(old_corpus_backup)
            corpus_moved = True
        staging.rename(target)
        new_corpus_installed = True
        if log_path.exists():
            log_path.rename(old_log_backup)
            log_moved = True
        staged_log.rename(log_path)
    except Exception as exc:
        rollback_errors = []
        if new_corpus_installed and target.exists():
            try:
                target.rename(staging)
            except Exception as rollback_exc:
                rollback_errors.append(f"new corpus: {rollback_exc}")
        if corpus_moved and old_corpus_backup.exists():
            try:
                old_corpus_backup.rename(target)
            except Exception as rollback_exc:
                rollback_errors.append(f"old corpus: {rollback_exc}")
        if log_moved and old_log_backup.exists():
            try:
                old_log_backup.rename(log_path)
            except Exception as rollback_exc:
                rollback_errors.append(f"old log: {rollback_exc}")
        if rollback_errors:
            raise RuntimeError(
                f"Switch failed: {exc}; rollback incomplete: {rollback_errors}; "
                "inspect backup/staging directories before running again."
            ) from exc
        raise RuntimeError(f"Switch failed; previous corpus/log restored: {exc}") from exc


    return target


def main_top1_corpus(config_path):
    return build_report_models_source_ablation_top1_corpus(load_config(config_path))
