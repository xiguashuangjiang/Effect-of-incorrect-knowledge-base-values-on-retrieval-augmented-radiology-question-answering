"""
rar_pipeline/local_search.py
Created on October 13, 2026

@author: Yangtingting Wang
https://github.com/xiguashuangjiang
"""

from pathlib import Path
import ast
import hashlib
import os
import random
import re
from functools import lru_cache
from typing import List
from copy import deepcopy
from contextvars import ContextVar

from langchain_core.tools import tool
from rank_bm25 import BM25Okapi


RETRIEVAL_LOG = ContextVar(
    "retrieval_log",
    default=None,
)

RETRIEVAL_STAGE = ContextVar(
    "retrieval_stage",
    default="unknown"
)

RETRIEVAL_SECTION = ContextVar(
    "retrieval_section",
    default=None
)


DEFAULT_CORPUS_DIR = (
    Path(__file__).resolve().parent.parent
    / "corpus"
    / "clean"
)

CORPUS_DIR = Path(
    os.getenv("LOCAL_CORPUS_DIR", str(DEFAULT_CORPUS_DIR))
).resolve()


def set_corpus_dir(path: Path | str) -> None:
    global CORPUS_DIR
    CORPUS_DIR = Path(path).resolve()

    if not CORPUS_DIR.exists():
        raise FileNotFoundError(
            f"Corpus directory does not exist: {CORPUS_DIR}"
        )


VOLUME_MODE = False
VOLUME_TOP_K = 5
VOLUME_PADDING_RATIO = 0.0
VOLUME_PADDING_SEED = 20260909

ALLOWED_VOLUME_TOP_K = {1, 3, 5, 10, 20}
ALLOWED_PADDING_RATIOS = {0.0, 0.50, 0.75, 0.90, 0.95}


def configure_volume_mode(
    enabled: bool = False,
    top_k: int = 5,
    padding_ratio: float = 0.0,
    seed: int = 20260909,
) -> None:
    global VOLUME_MODE
    global VOLUME_TOP_K
    global VOLUME_PADDING_RATIO
    global VOLUME_PADDING_SEED

    enabled = bool(enabled)
    top_k = int(top_k)
    padding_ratio = float(padding_ratio)
    seed = int(seed)

    if enabled:
        if top_k not in ALLOWED_VOLUME_TOP_K:
            raise ValueError(
                f"top_k must be one of {sorted(ALLOWED_VOLUME_TOP_K)}, got {top_k}"
            )

        if padding_ratio not in ALLOWED_PADDING_RATIOS:
            raise ValueError(
                "padding_ratio must be one of "
                f"{sorted(ALLOWED_PADDING_RATIOS)}, got {padding_ratio}"
            )

        if padding_ratio > 0.0 and top_k != 5:
            raise ValueError(
                "Needle-in-haystack conditions keep top_k fixed at 5."
            )

    VOLUME_MODE = enabled
    VOLUME_TOP_K = top_k
    VOLUME_PADDING_RATIO = padding_ratio
    VOLUME_PADDING_SEED = seed


def clear_retrieval_log():
    RETRIEVAL_LOG.set([])


def get_retrieval_log():
    current_log = RETRIEVAL_LOG.get()
    return deepcopy(current_log) if current_log is not None else []


def _append_retrieval_log(record: dict) -> None:
    current_log = RETRIEVAL_LOG.get()
    if current_log is None:
        current_log = []
        RETRIEVAL_LOG.set(current_log)
    current_log.append(record)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


@lru_cache(maxsize=8)
def load_bm25_index(corpus_dir_str: str):
    corpus_dir = Path(corpus_dir_str)
    files = sorted(corpus_dir.glob("*.txt"))

    if not files:
        raise FileNotFoundError(
            f"No .txt files found in corpus directory: {corpus_dir}"
        )

    filenames = []
    documents = []
    tokenized_documents = []

    for path in files:
        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore"
            ).strip()
        except Exception:
            continue

        if not text:
            continue

        filenames.append(path.name)
        documents.append(text)
        tokenized_documents.append(tokenize(text))

    if not documents:
        raise RuntimeError(
            f"Corpus was found but no readable documents were loaded from {corpus_dir}"
        )

    bm25 = BM25Okapi(tokenized_documents)


    return bm25, filenames, documents


def _normalize_queries(queries: List[str] | str) -> List[str]:
    if isinstance(queries, str):
        try:
            parsed = ast.literal_eval(queries)

            if isinstance(parsed, list):
                return [str(q) for q in parsed]

            return [queries]

        except (ValueError, SyntaxError):
            return [queries]

    return [str(q) for q in queries]


def _padding_count(relevant_count: int, ratio: float) -> int:
    if ratio <= 0.0:
        return 0

    return int(round(
        relevant_count * ratio / (1.0 - ratio)
    ))


def _stable_query_seed(query: str, base_seed: int) -> int:
    digest = hashlib.sha256(
        f"{base_seed}|{query}".encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        "big",
        signed=False,
    )


def _select_irrelevant_indices(
    scores,
    relevant_indices: list[int],
    n_padding: int,
    query: str,
) -> list[int]:
    if n_padding <= 0:
        return []

    relevant_set = set(relevant_indices)

    zero_score_pool = [
        idx
        for idx, score in enumerate(scores)
        if idx not in relevant_set
        and float(score) <= 0.0
    ]

    rng = random.Random(
        _stable_query_seed(
            query,
            VOLUME_PADDING_SEED,
        )
    )

    if len(zero_score_pool) >= n_padding:
        return rng.sample(
            zero_score_pool,
            n_padding,
        )

    raise RuntimeError(f"Need {n_padding} articles with BM25 score at most zero. Found {len(zero_score_pool)}.")


@tool
def local_corpus_search(
    queries: List[str] | str,
    max_results: int = 10,
) -> str:
    """
    Search the local Radiopaedia corpus using BM25.

    LEGACY MODE (default):
        Behaves as before. max_results supplied by the tool call controls
        the number of BM25 results per query, with default=10.

    VOLUME MODE:
        The externally configured top_k controls the number of BM25 results.
        The tool-call max_results value is ignored for retrieval and is only
        recorded in the log. Optional irrelevant padding is then appended.

    Returns:
        A formatted text bundle containing Radiopaedia documents.
    """
    queries = _normalize_queries(queries)

    bm25, filenames, documents = load_bm25_index(
        str(CORPUS_DIR)
    )

    output_parts = []
    seen_articles = set()

    for query in queries:
        query = query.strip()

        if not query:
            continue

        scores = bm25.get_scores(
            tokenize(query)
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        if VOLUME_MODE:
            effective_top_k = VOLUME_TOP_K
        else:
            effective_top_k = max_results

        top_indices = ranked_indices[:effective_top_k]

        padding_indices = []

        if VOLUME_MODE and VOLUME_PADDING_RATIO > 0.0:
            n_padding = _padding_count(
                len(top_indices),
                VOLUME_PADDING_RATIO,
            )

            padding_indices = _select_irrelevant_indices(
                scores=scores,
                relevant_indices=top_indices,
                n_padding=n_padding,
                query=query,
            )

        retrieved_articles = []

        for rank, idx in enumerate(
            top_indices,
            start=1,
        ):
            item = {
                "rank": rank,
                "article_id": filenames[idx],
                "score": float(scores[idx]),
            }

            if VOLUME_MODE:
                item["retrieval_role"] = "relevant"

            retrieved_articles.append(item)

        for pad_rank, idx in enumerate(
            padding_indices,
            start=1,
        ):
            retrieved_articles.append({
                "rank": len(top_indices) + pad_rank,
                "article_id": filenames[idx],
                "score": float(scores[idx]),
                "retrieval_role": "irrelevant_padding",
            })

        log_record = {
            "stage": RETRIEVAL_STAGE.get(),
            "section": RETRIEVAL_SECTION.get(),
            "query": query,
            "results": retrieved_articles,
        }

        if VOLUME_MODE:
            log_record.update({
                "retrieval_mode": "volume",
                "configured_top_k": VOLUME_TOP_K,
                "tool_requested_max_results": max_results,
                "padding_ratio": VOLUME_PADDING_RATIO,
                "padding_seed": VOLUME_PADDING_SEED,
                "n_relevant": len(top_indices),
                "n_padding": len(padding_indices),
                "n_total": (
                    len(top_indices)
                    + len(padding_indices)
                ),
            })

        _append_retrieval_log(log_record)

        output_parts.append(
            f"Search query: {query}"
        )

        combined_indices = (
            top_indices
            + padding_indices
        )

        delivered_relevant = 0
        delivered_padding = 0

        for rank, idx in enumerate(
            combined_indices,
            start=1,
        ):
            article_id = filenames[idx]

            if article_id in seen_articles:
                if VOLUME_MODE:
                    retrieved_articles[rank - 1]["included_in_context"] = False
                continue

            seen_articles.add(article_id)

            if VOLUME_MODE:
                retrieved_articles[rank - 1]["included_in_context"] = True

                if rank <= len(top_indices):
                    delivered_relevant += 1
                else:
                    delivered_padding += 1

            text = documents[idx]

            preview = text[:2000]

            output_parts.append(
                "\n"
                + "=" * 80
                + "\n"
                + f"Result {rank}\n"
                + f"Source file: {article_id}\n"
                + f"BM25 score: {scores[idx]:.3f}\n"
                + "-" * 80
                + "\n"
                + preview
            )

        if VOLUME_MODE:
            log_record.update({
                "n_delivered_relevant": delivered_relevant,
                "n_delivered_padding": delivered_padding,
                "n_delivered_total": delivered_relevant + delivered_padding,
            })

        output_parts.append("\n")

    if not output_parts:
        return "No valid search query was provided."

    return "\n".join(output_parts)
