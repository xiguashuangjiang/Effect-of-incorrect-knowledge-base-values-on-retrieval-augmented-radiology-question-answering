"""
src/rebuild_coverage_corpora.py
Created on November 6, 2026

@author: Yangtingting Wang
https://github.com/xiguashuangjiang
"""

from __future__ import annotations
import hashlib, json, math, random, shutil
from pathlib import Path
from src.config import load_config

LEVELS = (10, 25, 50, 100)

def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def qkey(row):
    return str(row["dataset"]), int(row["q_number"])

def validate_clean_reports(rows):
    if len(rows) != 200:
        raise RuntimeError(f"Expected 200 clean reports, got {len(rows)}.")
    keys = [qkey(r) for r in rows]
    if len(set(keys)) != 200:
        raise RuntimeError("Clean reports do not contain 200 unique (dataset, q_number) keys.")
    bad = [k for k, r in zip(keys, rows) if r.get("report_status") != "complete"]
    if bad:
        raise RuntimeError(f"Clean reports contain {len(bad)} failed/incomplete records: {bad[:10]}")

def extract_pool(rows):
    stats = {}
    for row in rows:
        dataset, qnum = qkey(row)
        seen_q = set()
        for entry in row.get("retrieval_log") or []:
            for result in entry.get("results") or []:
                aid = str(result.get("article_id") or "").strip()
                if not aid:
                    continue
                s = stats.setdefault(aid, {
                    "article_id": aid,
                    "retrieval_occurrences": 0,
                    "question_ids": [],
                })
                s["retrieval_occurrences"] += 1
                seen_q.add(aid)
        for aid in sorted(seen_q):
            stats[aid]["question_ids"].append(
                {"dataset": dataset, "q_number": qnum}
            )
    return sorted(stats), stats

def deletion_count(n, level):
    if level == 100:
        return n
    return int(math.floor(n * (level / 100.0) + 0.5))

def build_nested(pool, seed):
    ordered = list(pool)
    random.Random(seed).shuffle(ordered)
    return {level: ordered[:deletion_count(len(ordered), level)] for level in LEVELS}

def rebuild(clean_files, outdir, deleted):
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, src in clean_files.items():
        if name not in deleted:
            shutil.copy2(src, outdir / name)

def qc(clean_files, outdir, deleted):
    outfiles = {p.name: p for p in outdir.glob("*.txt")}
    expected = set(clean_files) - deleted
    actual = set(outfiles)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    still_present = sorted(deleted & actual)
    mismatches = []
    for name in sorted(expected & actual):
        if sha256_file(clean_files[name]) != sha256_file(outfiles[name]):
            mismatches.append(name)
    ok = not (missing or unexpected or still_present or mismatches)
    return {
        "status": "PASS" if ok else "FAIL",
        "expected_files": len(expected),
        "actual_files": len(actual),
        "deleted_files": len(deleted),
        "missing_expected": missing,
        "unexpected_files": unexpected,
        "deleted_still_present": still_present,
        "hash_mismatches": mismatches,
    }

def main_coverage_corpora(config_path):
    cfg = load_config(config_path)

    seed = int(cfg.get("coverage_rebuild", {}).get("seed", 20260907))
    clean_report_path = Path(
        cfg.get("coverage_rebuild", {}).get(
            "source_clean_reports",
            Path(cfg["outputs"]["formal_reports"]) / "clean.json"
        )
    )
    clean_dir = Path(cfg["corpora"]["clean"])
    output_corpora = cfg.get("coverage_rebuild", {}).get(
        "output_corpora",
        cfg["corpora"]["coverage"]
    )
    log_root = Path(cfg["paths"]["output_root"]) / "coverage_rebuild"
    log_root.mkdir(parents=True, exist_ok=True)

    if not clean_report_path.exists():
        raise FileNotFoundError(
            f"New full clean report file not found:\n{clean_report_path}\n"
            "Wait until clean 200 has finished."
        )

    rows = load_json(clean_report_path)
    if not isinstance(rows, list):
        raise RuntimeError("Clean report JSON must be a list.")
    validate_clean_reports(rows)

    pool, stats = extract_pool(rows)
    if not pool:
        raise RuntimeError("No retrieved article IDs found in clean retrieval logs.")

    clean_files = {p.name: p for p in clean_dir.glob("*.txt")}
    if not clean_files:
        raise RuntimeError(f"No .txt files in clean corpus: {clean_dir}")

    missing = [x for x in pool if x not in clean_files]
    if missing:
        raise RuntimeError(
            f"{len(missing)} retrieved article IDs are not exact clean-corpus filenames. "
            f"First few: {missing[:10]}"
        )

    deletions = build_nested(pool, seed)

    if not (
        set(deletions[10]) <= set(deletions[25]) <=
        set(deletions[50]) <= set(deletions[100])
    ):
        raise RuntimeError("Nested deletion QC failed.")
    if set(deletions[100]) != set(pool):
        raise RuntimeError("100% deletion set does not equal the full relevant pool.")

    pool_rows = []
    for aid in pool:
        pool_rows.append({
            "article_id": aid,
            "corpus_filename": aid,
            "relative_path": aid,
            "question_ids": stats[aid]["question_ids"],
            "retrieval_occurrences": stats[aid]["retrieval_occurrences"],
            "sha256": sha256_file(clean_files[aid]),
        })

    save_json(log_root / "coverage_relevant_articles.json", {
        "source_clean_reports": str(clean_report_path),
        "seed": seed,
        "clean_corpus_files": len(clean_files),
        "relevant_pool_size": len(pool),
        "articles": pool_rows,
    })

    summary = {
        "source_clean_reports": str(clean_report_path),
        "seed": seed,
        "relevant_pool_size": len(pool),
        "levels": {},
    }


    for level in LEVELS:
        cond = f"cov{level:03d}"
        outdir = Path(output_corpora[cond])
        deleted_list = deletions[level]
        deleted = set(deleted_list)

        rebuild(clean_files, outdir, deleted)

        deletion_records = []
        for i, aid in enumerate(deleted_list, start=1):
            deletion_records.append({
                "deletion_order": i,
                "article_id": aid,
                "corpus_filename": aid,
                "relative_path": aid,
                "question_ids": stats[aid]["question_ids"],
                "retrieval_occurrences": stats[aid]["retrieval_occurrences"],
            })

        save_json(log_root / f"coverage_{level:03d}_deletions.json", {
            "level_percent": level,
            "seed": seed,
            "relevant_pool_size": len(pool),
            "n_deleted": len(deleted_list),
            "deleted_articles": deletion_records,
        })

        result = qc(clean_files, outdir, deleted)
        save_json(log_root / f"coverage_{level:03d}_qc.json", result)

        summary["levels"][cond] = {
            "output_dir": str(outdir),
            "n_deleted": len(deleted),
            "n_remaining": result["actual_files"],
            "qc_status": result["status"],
        }

        if result["status"] != "PASS":
            raise RuntimeError(f"{cond} QC failed.")

    save_json(log_root / "coverage_rebuild_summary.json", summary)
