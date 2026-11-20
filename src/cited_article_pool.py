"""
src/cited_article_pool.py
Created on October 26, 2026

@author: Yangtingting Wang
https://github.com/xiguashuangjiang
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


TXT_TOKEN_RE = re.compile(
    r"""(?ix)
    [A-Za-z0-9]
    [A-Za-z0-9._-]*
    \.txt
    """
)


HASH_RE = re.compile(
    r"__([0-9a-fA-F]{10})(?:\.|$)"
)

STOP_HEADING_RE = re.compile(
    r"""(?ix)^\s*\#{0,6}\s*
    (?:
        option\s+[A-E]\b
        | final\s+note\b
        | supporting(?:\s+evidence)?\b
        | contradictory(?:\s+evidence)?\b
        | neutral\s+synthesis\b
        | synthesis\b
        | name\b
    )
    """
)

SOURCES_HEADING_RE = re.compile(
    r"(?i)^\s*#{0,6}\s*sources?\s*:?[ \t]*(.*)$"
)


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _report_text(record: dict) -> str:
    for key in (
        "report",
        "deep_research_report",
        "rar_report",
        "content",
        "text",
    ):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _normalize_source_token(token: str) -> str:
    token = str(token).strip()
    token = token.strip("`*_[](){}<>\"'")
    token = token.replace("\\", "/")
    token = token.rsplit("/", 1)[-1]

    token = re.sub(
        r"^\s*\d+\s*[.)]\s*",
        "",
        token,
    )

    return token.strip(" ,;:.")


def extract_sources_from_report(report_text: str) -> list[str]:
    lines = str(report_text or "").splitlines()
    found: list[str] = []
    in_sources = False

    for line in lines:
        source_match = SOURCES_HEADING_RE.match(line)

        if source_match:
            in_sources = True
            payload = source_match.group(1)
            for match in TXT_TOKEN_RE.findall(payload):
                token = _normalize_source_token(match)
                if token.lower().endswith(".txt"):
                    found.append(token)
            continue

        if in_sources and STOP_HEADING_RE.match(line):
            in_sources = False

        if not in_sources:
            continue

        for match in TXT_TOKEN_RE.findall(line):
            token = _normalize_source_token(match)
            if token.lower().endswith(".txt"):
                found.append(token)

    return found


def _canonical_article_name(name: str) -> str:
    name = Path(str(name)).name
    lowered = name.lower()

    suffixes = (
        ".clean.round2.round3.round4.txt",
        ".clean.round2.round3.txt",
        ".clean.round2.txt",
        ".clean.txt",
        ".round2.round3.round4.txt",
        ".round2.round3.txt",
        ".round2.txt",
    )

    for suffix in suffixes:
        if lowered.endswith(suffix):
            return name[: -len(suffix)] + ".txt"

    return name


def build_corpus_index(
    clean_corpus: str | Path,
) -> tuple[dict[str, Path], dict[str, Path]]:
    clean_corpus = Path(clean_corpus)

    if not clean_corpus.exists():
        raise FileNotFoundError(
            f"Clean corpus not found: {clean_corpus}"
        )

    paths = sorted(clean_corpus.rglob("*.txt"))

    if not paths:
        raise RuntimeError(
            f"No .txt files found under {clean_corpus}"
        )

    name_index: dict[str, Path] = {}
    hash_index: dict[str, Path] = {}

    for path in paths:
        aliases = {
            path.name,
            path.name.lower(),
            _canonical_article_name(path.name),
            _canonical_article_name(path.name).lower(),
        }

        for alias in aliases:
            old = name_index.get(alias)

            if old is not None and old != path:
                raise RuntimeError(
                    "Ambiguous clean-corpus article alias:\n"
                    f"  alias: {alias}\n"
                    f"  first: {old}\n"
                    f"  second: {path}"
                )

            name_index[alias] = path

        match = HASH_RE.search(path.name)

        if match:
            article_hash = match.group(1).lower()
            old = hash_index.get(article_hash)

            if old is not None and old != path:
                raise RuntimeError(
                    "Ambiguous clean-corpus article hash:\n"
                    f"  hash: {article_hash}\n"
                    f"  first: {old}\n"
                    f"  second: {path}"
                )

            hash_index[article_hash] = path

    return name_index, hash_index

def _article_title_key(filename: str) -> str | None:
    normalized = _canonical_article_name(_normalize_source_token(filename))
    match = re.fullmatch(r"(.+?)__[0-9a-fA-F]{10}\.txt", normalized, flags=re.I)
    return match.group(1).casefold() if match else None


def resolve_article(
    source_id: str,
    corpus_index: dict[str, Path],
    hash_index: dict[str, Path],
    *,
    resolution_log: list[dict] | None = None,
) -> Path:
    source_id = _normalize_source_token(source_id)
    candidates = (
        source_id, source_id.lower(),
        _canonical_article_name(source_id),
        _canonical_article_name(source_id).lower(),
    )
    for candidate in candidates:
        path = corpus_index.get(candidate)
        if path is not None:
            return path

    match = HASH_RE.search(source_id)
    if match:
        path = hash_index.get(match.group(1).lower())
        if path is not None:
            return path

    title = _article_title_key(source_id)
    if title is not None:
        matches = sorted(
            {path for path in corpus_index.values()
             if _article_title_key(path.name) == title},
            key=lambda p: str(p),
        )
        if len(matches) == 1:
            chosen = matches[0]
            if resolution_log is not None:
                resolution_log.append({
                    "resolution_method": "unique_exact_title_hash_mismatch",
                    "source_id": source_id,
                    "resolved_article": chosen.name,
                    "source_hash": match.group(1).lower() if match else None,
                    "resolved_hash": (
                        HASH_RE.search(chosen.name).group(1).lower()
                        if HASH_RE.search(chosen.name) else None
                    ),
                })
            return chosen
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous exact article title for {source_id}: "
                + ", ".join(str(p) for p in matches)
            )

    raise KeyError(f"Could not resolve cited article in clean corpus: {source_id}")


def _report_models_source_ablation_root(config: dict) -> Path:
    return Path(
        config.get("outputs", {}).get(
            "report_models_source_ablation",
            Path(config["paths"]["output_root"]) / "report_models_source_ablation",
        )
    )
