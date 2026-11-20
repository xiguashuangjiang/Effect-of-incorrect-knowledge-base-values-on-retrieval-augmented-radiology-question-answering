"""
src/direct_rag.py
Created on November 12, 2026

@author: Yangtingting Wang
https://github.com/xiguashuangjiang
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any
from pathlib import Path
import hashlib

from rank_bm25 import BM25Okapi

from src.config import load_config


TOKEN_PATTERN = r"[A-Za-z0-9]+"
TOKEN_RE = re.compile(TOKEN_PATTERN)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _canonical_json_sha256(obj: Any) -> str:
    payload = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _read_questions(path: Path, expected_n: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise RuntimeError("Benchmark runtime JSON must contain a list.")

    if len(data) != expected_n:
        raise RuntimeError(
            f"Expected {expected_n} benchmark questions, got {len(data)}."
        )

    return data

def _question_stem(question: dict) -> str:
    for key in ("question", "question_stem", "stem", "summary"):
        value = question.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    raise KeyError(
        "Could not find question stem in one of: "
        "'question', 'question_stem', 'stem', 'summary'. "
        f"Record identifiers: dataset={question.get('dataset')!r}, "
        f"q_number={question.get('q_number')!r}"
    )

def _article_id(path: Path) -> str:
    return path.stem


@dataclass
class CorpusIndex:
    condition: str
    corpus_path: Path
    filenames: list[str]
    article_ids: list[str]
    texts: list[str]
    tokenized: list[list[str]]
    source_sha256: list[str]
    empty_articles: list[str]
    corpus_fingerprint: str
    bm25: BM25Okapi


def _fingerprint_records(
    filenames: list[str],
    source_sha256: list[str],
) -> str:
    h = hashlib.sha256()
    for filename, file_hash in zip(filenames, source_sha256):
        h.update(filename.encode("utf-8"))
        h.update(b"\t")
        h.update(file_hash.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _load_corpus_source(corpus_path: Path) -> tuple[
    list[str],
    list[str],
    list[str],
    list[list[str]],
    list[str],
    list[str],
    str,
]:
    if not corpus_path.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_path}")

    files = sorted(
        (p for p in corpus_path.iterdir() if p.is_file() and p.suffix.lower() == ".txt"),
        key=lambda p: p.name,
    )
    if not files:
        raise RuntimeError(f"No .txt articles found in corpus: {corpus_path}")

    filenames: list[str] = []
    article_ids: list[str] = []
    texts: list[str] = []
    tokenized: list[list[str]] = []
    source_sha256: list[str] = []
    empty_articles: list[str] = []

    for path in files:
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(
                f"READ FAILURE: {path}. Index construction stopped."
            ) from exc

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                f"UTF-8 DECODE FAILURE: {path}. Index construction stopped."
            ) from exc

        stripped = text.strip()
        if not stripped:
            empty_articles.append(path.name)

        filenames.append(path.name)
        article_ids.append(_article_id(path))
        texts.append(text)
        tokenized.append(_tokenize(text))
        source_sha256.append(_sha256_bytes(raw))

    fingerprint = _fingerprint_records(filenames, source_sha256)
    return (
        filenames,
        article_ids,
        texts,
        tokenized,
        source_sha256,
        empty_articles,
        fingerprint,
    )


def _index_cache_key(
    *,
    corpus_fingerprint: str,
    rank_bm25_version: str,
    k1: float,
    b: float,
    epsilon: float,
    tokenizer_pattern: str,
) -> str:
    return _canonical_json_sha256(
        {
            "corpus_fingerprint": corpus_fingerprint,
            "rank_bm25_version": rank_bm25_version,
            "implementation": "rank_bm25.BM25Okapi",
            "k1": k1,
            "b": b,
            "epsilon": epsilon,
            "tokenizer_pattern": tokenizer_pattern,
            "lowercase": True,
        }
    )


def _build_or_load_index(
    *,
    condition: str,
    corpus_path: Path,
    cache_dir: Path,
    k1: float,
    b: float,
    epsilon: float,
    rank_bm25_version: str,
) -> tuple[CorpusIndex, Path, bool]:
    (
        filenames,
        article_ids,
        texts,
        tokenized,
        source_sha256,
        empty_articles,
        corpus_fingerprint,
    ) = _load_corpus_source(corpus_path)

    key = _index_cache_key(
        corpus_fingerprint=corpus_fingerprint,
        rank_bm25_version=rank_bm25_version,
        k1=k1,
        b=b,
        epsilon=epsilon,
        tokenizer_pattern=TOKEN_PATTERN,
    )
    cache_path = cache_dir / f"index_{condition}_{key}.pkl"

    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cached = pickle.load(f)

            if (
                cached.get("cache_key") == key
                and cached.get("filenames") == filenames
                and cached.get("source_sha256") == source_sha256
            ):
                bm25 = cached["bm25"]
                return (
                    CorpusIndex(
                        condition=condition,
                        corpus_path=corpus_path,
                        filenames=filenames,
                        article_ids=article_ids,
                        texts=texts,
                        tokenized=tokenized,
                        source_sha256=source_sha256,
                        empty_articles=empty_articles,
                        corpus_fingerprint=corpus_fingerprint,
                        bm25=bm25,
                    ),
                    cache_path,
                    True,
                )
        except Exception:
            pass

    bm25 = BM25Okapi(
        tokenized,
        k1=k1,
        b=b,
        epsilon=epsilon,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(
            {
                "cache_key": key,
                "filenames": filenames,
                "source_sha256": source_sha256,
                "bm25": bm25,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return (
        CorpusIndex(
            condition=condition,
            corpus_path=corpus_path,
            filenames=filenames,
            article_ids=article_ids,
            texts=texts,
            tokenized=tokenized,
            source_sha256=source_sha256,
            empty_articles=empty_articles,
            corpus_fingerprint=corpus_fingerprint,
            bm25=bm25,
        ),
        cache_path,
        False,
    )


def _evidence_cache_key(
    *,
    index_cache_key_inputs_sha256: str,
    query: str,
    top_k: int,
    passage_chars: int,
    evidence_char_cap: int,
) -> str:
    return _canonical_json_sha256(
        {
            "index_inputs_sha256": index_cache_key_inputs_sha256,
            "query": query,
            "top_k": top_k,
            "passage_chars": passage_chars,
            "evidence_char_cap": evidence_char_cap,
            "format": "[Source {rank}: {article_id}]\\n{text}",
            "separator": "\\n\\n",
        }
    )


def _rank_top_k(index: CorpusIndex, query: str, top_k: int) -> list[tuple[int, float]]:
    query_tokens = _tokenize(query)
    scores = index.bm25.get_scores(query_tokens)

    ranked = sorted(
        range(len(index.filenames)),
        key=lambda i: (-float(scores[i]), index.filenames[i]),
    )
    selected = ranked[:top_k]

    if len(selected) != top_k:
        raise RuntimeError(
            f"Could not retrieve {top_k} articles; corpus has only {len(selected)}."
        )

    return [(i, float(scores[i])) for i in selected]


def _make_evidence_record(
    *,
    question: dict[str, Any],
    index: CorpusIndex,
    index_inputs_sha256: str,
    top_k: int,
    passage_chars: int,
    evidence_char_cap: int,
) -> dict[str, Any]:
    query = _question_stem(question)
    ranked = _rank_top_k(index, query, top_k)

    selected_passages: list[dict[str, Any]] = []
    blocks: list[str] = []

    for rank, (i, score) in enumerate(ranked, start=1):
        stripped = index.texts[i].strip()
        passage = stripped[:passage_chars]
        header = f"[Source {rank}: {index.article_ids[i]}]"
        block = f"{header}\n{passage}"

        selected_passages.append(
            {
                "rank": rank,
                "article_id": index.article_ids[i],
                "filename": index.filenames[i],
                "bm25_score": score,
                "source_sha256": index.source_sha256[i],
                "full_article_chars": len(index.texts[i]),
                "stripped_article_chars": len(stripped),
                "selected_passage": passage,
                "selected_passage_chars": len(passage),
                "block_text": block,
            }
        )
        blocks.append(block)

    full_evidence = "\n\n".join(blocks)
    included_evidence = full_evidence[:evidence_char_cap]
    truncated = len(full_evidence) > evidence_char_cap

    cursor = 0
    for j, item in enumerate(selected_passages):
        block = item["block_text"]
        prefix = "" if j == 0 else "\n\n"
        start = cursor + len(prefix)
        end = start + len(block)
        overlap_start = min(max(start, 0), len(included_evidence))
        overlap_end = min(max(end, 0), len(included_evidence))
        item["included_block_text"] = (
            included_evidence[overlap_start:overlap_end]
            if overlap_end > overlap_start
            else ""
        )
        item["included_block_chars"] = len(item["included_block_text"])
        item["truncated_by_common_cap"] = item["included_block_chars"] < len(block)
        cursor = end

    evidence_key = _evidence_cache_key(
        index_cache_key_inputs_sha256=index_inputs_sha256,
        query=query,
        top_k=top_k,
        passage_chars=passage_chars,
        evidence_char_cap=evidence_char_cap,
    )

    return {
        "dataset": question.get("dataset"),
        "q_number": question.get("q_number"),
        "condition": index.condition,
        "query_text": query,
        "query_sha256": _sha256_text(query),
        "evidence_cache_key": evidence_key,
        "selected_passages": selected_passages,
        "full_evidence_text": full_evidence,
        "full_evidence_chars": len(full_evidence),
        "evidence_text": included_evidence,
        "evidence_chars": len(included_evidence),
        "evidence_sha256": _sha256_text(included_evidence),
        "truncated": truncated,
        "truncated_chars": len(full_evidence) - len(included_evidence),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _load_jsonl_by_id(path: Path) -> dict[tuple[Any, Any], dict[str, Any]]:
    out: dict[tuple[Any, Any], dict[str, Any]] = {}
    if not path.exists():
        return out

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record.get("dataset"), record.get("q_number"))
            if key in out:
                raise RuntimeError(f"Duplicate cached evidence record {key} in {path}")
            out[key] = record
    return out


def prepare_direct_rag_evidence(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config["direct_rag"]

    benchmark_path = Path(config["benchmark"]["runtime_json"])
    expected_n = int(config["benchmark"].get("expected_total_questions", 200))

    corpus_paths = {
        "rag_clean": Path(config["corpora"]["clean"]),
        "rag_num100": Path(config["corpora"]["numerical"]["num100"]),
    }

    output_dir = Path(cfg["outputs"]["evidence_dir"])
    settings_path = Path(cfg["outputs"]["settings_json"])
    manifest_path = Path(cfg["outputs"]["input_manifest_json"])
    cache_dir = Path(cfg["outputs"]["cache_dir"])

    k1 = float(cfg["bm25"]["k1"])
    b = float(cfg["bm25"]["b"])
    epsilon = float(cfg["bm25"]["epsilon"])
    top_k = int(cfg["top_k"])
    passage_chars = int(cfg["passage_chars"])
    evidence_char_cap = int(cfg["evidence_char_cap"])

    expected = {
        "k1": 1.5,
        "b": 0.75,
        "epsilon": 0.25,
        "top_k": 5,
        "passage_chars": 2000,
        "evidence_char_cap": 12000,
    }
    actual = {
        "k1": k1,
        "b": b,
        "epsilon": epsilon,
        "top_k": top_k,
        "passage_chars": passage_chars,
        "evidence_char_cap": evidence_char_cap,
    }
    if actual != expected:
        raise RuntimeError(
            "direct_rag config does not match the direct-retrieval settings.\n"
            f"Expected: {expected}\nActual:   {actual}"
        )

    rank_bm25_version = package_version("rank-bm25")
    questions = _read_questions(benchmark_path, expected_n)
    benchmark_sha256 = _sha256_bytes(benchmark_path.read_bytes())

    run_settings = {
        "run_type": "direct_retrieval_evidence",
        "implementation": "rank_bm25.BM25Okapi",
        "rank_bm25_version": rank_bm25_version,
        "tokenizer_pattern": TOKEN_PATTERN,
        "tokenizer_lowercase": True,
        "bm25": {"k1": k1, "b": b, "epsilon": epsilon},
        "top_k": top_k,
        "passage_chars": passage_chars,
        "evidence_char_cap": evidence_char_cap,
        "query_source": "question stem only",
        "ranking": "decreasing BM25 score; ties by filename",
        "retrieve_zero_scores": True,
        "passage_format": "[Source {rank}: {article_id}]\\n{text}",
        "passage_separator": "\\n\\n",
        "truncate_rule": "common evidence only; truncate from end",
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": benchmark_sha256,
    }
    retrieval_config_sha256 = _canonical_json_sha256(run_settings)
    run_settings["retrieval_config_sha256"] = retrieval_config_sha256

    summary: dict[str, Any] = {
        "retrieval_config_sha256": retrieval_config_sha256,
        "conditions": {},
    }
    manifest: dict[str, Any] = {
        "run_type": "direct_retrieval_evidence",
        "retrieval_config_sha256": retrieval_config_sha256,
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": benchmark_sha256,
        "conditions": {},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for condition, corpus_path in corpus_paths.items():
        index, index_cache_path, index_cache_hit = _build_or_load_index(
            condition=condition,
            corpus_path=corpus_path,
            cache_dir=cache_dir,
            k1=k1,
            b=b,
            epsilon=epsilon,
            rank_bm25_version=rank_bm25_version,
        )

        index_inputs = {
            "corpus_fingerprint": index.corpus_fingerprint,
            "rank_bm25_version": rank_bm25_version,
            "implementation": "rank_bm25.BM25Okapi",
            "k1": k1,
            "b": b,
            "epsilon": epsilon,
            "tokenizer_pattern": TOKEN_PATTERN,
            "lowercase": True,
        }
        index_inputs_sha256 = _canonical_json_sha256(index_inputs)

        evidence_path = output_dir / f"{condition}.jsonl"
        existing = _load_jsonl_by_id(evidence_path)
        records: list[dict[str, Any]] = []
        evidence_cache_hits = 0

        for question in questions:
            key = (question.get("dataset"), question.get("q_number"))
            query = _question_stem(question)
            expected_cache_key = _evidence_cache_key(
                index_cache_key_inputs_sha256=index_inputs_sha256,
                query=query,
                top_k=top_k,
                passage_chars=passage_chars,
                evidence_char_cap=evidence_char_cap,
            )

            cached = existing.get(key)
            if (
                cached is not None
                and cached.get("condition") == condition
                and cached.get("evidence_cache_key") == expected_cache_key
                and cached.get("query_sha256") == _sha256_text(query)
                and cached.get("evidence_sha256")
                == _sha256_text(cached.get("evidence_text", ""))
            ):
                records.append(cached)
                evidence_cache_hits += 1
                continue

            records.append(
                _make_evidence_record(
                    question=question,
                    index=index,
                    index_inputs_sha256=index_inputs_sha256,
                    top_k=top_k,
                    passage_chars=passage_chars,
                    evidence_char_cap=evidence_char_cap,
                )
            )

        if len(records) != expected_n:
            raise RuntimeError(
                f"{condition}: expected {expected_n} evidence records, got {len(records)}."
            )

        ids = [(r["dataset"], r["q_number"]) for r in records]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"{condition}: duplicate (dataset, q_number) records.")

        _write_jsonl(evidence_path, records)
        evidence_file_sha256 = _sha256_bytes(evidence_path.read_bytes())

        condition_info = {
            "corpus_path": str(corpus_path),
            "corpus_fingerprint": index.corpus_fingerprint,
            "article_count": len(index.filenames),
            "empty_articles": index.empty_articles,
            "unreadable_articles": [],
            "index_inputs_sha256": index_inputs_sha256,
            "index_cache_path": str(index_cache_path),
            "index_cache_hit": index_cache_hit,
            "evidence_path": str(evidence_path),
            "evidence_file_sha256": evidence_file_sha256,
            "records": len(records),
            "evidence_cache_hits": evidence_cache_hits,
            "truncated_questions": sum(bool(r["truncated"]) for r in records),
            "max_full_evidence_chars": max(r["full_evidence_chars"] for r in records),
            "max_included_evidence_chars": max(r["evidence_chars"] for r in records),
        }
        summary["conditions"][condition] = condition_info
        manifest["conditions"][condition] = condition_info

    _write_json(settings_path, run_settings)
    _write_json(manifest_path, manifest)

    summary["settings_path"] = str(settings_path)
    summary["manifest_path"] = str(manifest_path)
    return summary


def main_direct_retrieval(config_path):
    return prepare_direct_rag_evidence(load_config(config_path))
