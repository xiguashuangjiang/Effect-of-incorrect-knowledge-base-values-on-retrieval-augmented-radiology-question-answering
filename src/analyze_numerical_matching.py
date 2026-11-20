"""
src/analyze_numerical_matching.py
Created on November 12, 2026

@author: Yangtingting Wang
https://github.com/xiguashuangjiang
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from functools import lru_cache
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from src.config import load_config


WS_RE = re.compile(r"\s+")
Q_RE = re.compile(r"(?:^|[^0-9])(?:q(?:uestion)?[_ -]?)?(\d{1,3})(?:[^0-9]|$)", re.I)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="strict")


def load_jsonish(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in (".jsonl", ".ndjson"):
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    raise ValueError(f"Unsupported structured file: {path}")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def normalize_ws(s: str) -> str:
    return WS_RE.sub(" ", s).strip()


def stable_pair_id(
    condition: str,
    dataset: str,
    q_number: Any,
    article_id: str,
    original_text: str,
    perturbed_text: str,
    source_start: Any,
    source_end: Any,
    entry_index: int,
) -> str:
    payload = "\t".join([
        str(condition),
        str(dataset),
        str(q_number),
        str(article_id),
        str(original_text),
        str(perturbed_text),
        str(source_start),
        str(source_end),
        str(entry_index),
    ])
    return sha256_text(payload)


def canonical_article_id(x: str) -> str:
    s = str(x).replace("\\", "/").split("/")[-1].strip()
    return s


def parse_q_number_from_name(path: Path) -> Optional[int]:
    m = Q_RE.search(path.stem)
    if not m:
        return None
    q = int(m.group(1))
    return q if 1 <= q <= 999 else None


@dataclass
class NormalizedText:
    text: str
    orig_start: List[int]
    orig_end: List[int]


@lru_cache(maxsize=8)
def normalize_ws_with_map(s: str) -> NormalizedText:
    out: List[str] = []
    starts: List[int] = []
    ends: List[int] = []

    i = 0
    n = len(s)
    while i < n and s[i].isspace():
        i += 1

    while i < n:
        if s[i].isspace():
            run_start = i
            while i < n and s[i].isspace():
                i += 1
            if i < n:
                out.append(" ")
                starts.append(run_start)
                ends.append(i)
        else:
            out.append(s[i])
            starts.append(i)
            ends.append(i + 1)
            i += 1

    return NormalizedText("".join(out), starts, ends)


def normalized_span_to_original(nt: NormalizedText, a: int, b: int) -> Tuple[int, int]:
    if a < 0 or b <= a or b > len(nt.text):
        raise ValueError((a, b, len(nt.text)))
    return nt.orig_start[a], nt.orig_end[b - 1]


def all_literal_spans(haystack: str, needle: str) -> List[Tuple[int, int]]:
    if not needle:
        return []
    out = []
    start = 0
    while True:
        i = haystack.find(needle, start)
        if i < 0:
            return out
        out.append((i, i + len(needle)))
        start = i + 1


_RANGE_CHARS = "-–—−"


def _first_nonspace_left(s: str, i: int) -> Tuple[Optional[str], int]:
    j = i - 1
    while j >= 0 and s[j].isspace():
        j -= 1
    return (s[j], j) if j >= 0 else (None, -1)


def _first_nonspace_right(s: str, i: int) -> Tuple[Optional[str], int]:
    j = i
    while j < len(s) and s[j].isspace():
        j += 1
    return (s[j], j) if j < len(s) else (None, len(s))


def is_complete_number_span(report_norm: str, a: int, b: int, value_norm: str) -> bool:
    if not value_norm or a < 0 or b <= a:
        return False

    if not re.search(r"\d", value_norm):
        return False

    left = report_norm[a - 1] if a > 0 else None
    right = report_norm[b] if b < len(report_norm) else None

    if left is not None and (left.isdigit() or left in ".,"):
        return False
    if right is not None and (right.isdigit() or (right in ".," and b + 1 < len(report_norm) and report_norm[b + 1].isdigit())):
        return False

    if left is not None and left in "+-" + _RANGE_CHARS:
        return False
    if right is not None and right in _RANGE_CHARS:
        return False

    lns, _ = _first_nonspace_left(report_norm, a)
    rns, ri = _first_nonspace_right(report_norm, b)
    bullet = re.search(r"(?:^|\n)[ \t]*[-+*][ \t]+$", report_norm[:a])
    if lns is not None and lns in _RANGE_CHARS and not bullet:
        return False
    if rns is not None and rns in _RANGE_CHARS and re.match(r"\s*\d", report_norm[ri + 1:]):
        return False
    if re.search(r"\d(?:\s*[A-Za-zµμ°%/]+)?\s+to\s+$", report_norm[:a], re.I):
        return False
    if re.match(r"\s+to\s*[-+]?\d", report_norm[b:], re.I):
        return False

    if right is not None and (right.isalpha() or right in "µμ°%"):
        return False

    if re.search(r"[A-Za-zµμ°%]$", value_norm):
        if right is not None and (right.isalnum() or right in "µμ°%"):
            return False

    if re.match(r"[-+]?(?:\d|\.\d)", value_norm):
        if left is not None and (left.isalnum() or left == "_"):
            return False

    return True


def complete_number_spans(report_text: str, value_text: str) -> Tuple[bool, List[Dict[str, int]], str, str]:
    nr = normalize_ws_with_map(report_text)
    nv = normalize_ws(value_text)
    spans = []
    for a, b in all_literal_spans(nr.text, nv):
        oa, ob = normalized_span_to_original(nr, a, b)
        if is_complete_number_span(report_text, oa, ob, nv):
            spans.append({
                "normalized_start": a,
                "normalized_end": b,
                "report_start": oa,
                "report_end": ob,
            })
    return bool(spans), spans, nr.text, nv


def substring_spans(report_text: str, value_text: str) -> Tuple[bool, List[Dict[str, int]], str, str]:
    nr = normalize_ws_with_map(report_text)
    nv = normalize_ws(value_text)
    spans = []
    for a, b in all_literal_spans(nr.text, nv):
        oa, ob = normalized_span_to_original(nr, a, b)
        spans.append({
            "normalized_start": a,
            "normalized_end": b,
            "report_start": oa,
            "report_end": ob,
        })
    return bool(spans), spans, nr.text, nv


ARTICLE_KEYS = (
    "article_id", "article", "corpus_filename", "filename", "file", "source_id",
    "document_id", "doc_id",
)
ORIGINAL_KEYS = (
    "original", "original_text", "old", "old_text", "source_text",
    "original_value", "old_value", "before",
)
PERTURBED_KEYS = (
    "perturbed", "perturbed_text", "new", "new_text", "replacement",
    "perturbed_value", "new_value", "after", "modified",
)
START_KEYS = (
    "source_start", "start", "start_offset", "char_start", "span_start",
    "original_start", "start_char",
)
END_KEYS = (
    "source_end", "end", "end_offset", "char_end", "span_end",
    "original_end", "end_char",
)


def first_present(d: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


@dataclass
class MutationEntry:
    article_id: str
    original: str
    perturbed: str
    source_start: Any
    source_end: Any
    entry_index: int
    raw: Dict[str, Any]


def _looks_like_mutation(d: Mapping[str, Any]) -> bool:
    return (
        first_present(d, ORIGINAL_KEYS) is not None
        and first_present(d, PERTURBED_KEYS) is not None
        and first_present(d, START_KEYS) is not None
        and first_present(d, END_KEYS) is not None
    )


def iter_mutation_entries(obj: Any, inherited_article: Optional[str] = None) -> Iterator[Tuple[str, Dict[str, Any]]]:
    if isinstance(obj, list):
        for item in obj:
            yield from iter_mutation_entries(item, inherited_article)
        return

    if not isinstance(obj, dict):
        return

    article = first_present(obj, ARTICLE_KEYS)
    if article is not None:
        inherited_article = str(article)

    if _looks_like_mutation(obj):
        if inherited_article is None:
            raise ValueError(f"Mutation entry missing article id: {obj}")
        yield inherited_article, dict(obj)
        return

    for k, v in obj.items():
        child_article = inherited_article
        if isinstance(v, list) and (k.endswith(".txt") or "__" in k):
            child_article = k
        yield from iter_mutation_entries(v, child_article)


def load_mutation_log(path: Path) -> Dict[str, List[MutationEntry]]:
    obj = load_jsonish(path)
    grouped_raw: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for article, raw in iter_mutation_entries(obj):
        grouped_raw[canonical_article_id(article)].append(raw)

    out: Dict[str, List[MutationEntry]] = {}
    for article, raws in grouped_raw.items():
        entries = []
        for idx, raw in enumerate(raws):
            original = first_present(raw, ORIGINAL_KEYS)
            perturbed = first_present(raw, PERTURBED_KEYS)
            start = first_present(raw, START_KEYS)
            end = first_present(raw, END_KEYS)
            if original is None or perturbed is None or start is None or end is None:
                continue
            entries.append(MutationEntry(
                article_id=article,
                original=str(original),
                perturbed=str(perturbed),
                source_start=start,
                source_end=end,
                entry_index=idx,
                raw=raw,
            ))
        out[article] = entries
    return out


REPORT_TEXT_KEYS = (
    "report", "report_text", "final_report", "evidence_report", "text",
    "content", "output",
)
DATASET_KEYS = ("dataset", "dataset_name")
Q_KEYS = ("q_number", "question_number", "question_id", "qid", "id")


@dataclass
class ReportRecord:
    condition: str
    dataset: str
    q_number: int
    report_text: str
    report_path: str
    report_sha256: str


def _iter_report_like(obj: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(obj, list):
        for x in obj:
            yield from _iter_report_like(x)
    elif isinstance(obj, dict):
        if any(k in obj for k in REPORT_TEXT_KEYS) and any(k in obj for k in Q_KEYS):
            yield obj
        else:
            for v in obj.values():
                yield from _iter_report_like(v)


def load_reports(path: Path, condition: str, default_dataset: str) -> Dict[Tuple[str, int], ReportRecord]:
    out: Dict[Tuple[str, int], ReportRecord] = {}

    def add(dataset: str, q: int, text: str, source: Path) -> None:
        key = (dataset, q)
        if key in out:
            raise ValueError(f"Duplicate report for {condition} {key}: {source}")
        out[key] = ReportRecord(
            condition=condition,
            dataset=dataset,
            q_number=q,
            report_text=text,
            report_path=str(source),
            report_sha256=sha256_text(text),
        )

    files: List[Path]
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in (".json", ".jsonl", ".ndjson", ".txt"))
    else:
        files = [path]

    for f in files:
        if f.suffix.lower() == ".txt":
            q = parse_q_number_from_name(f)
            if q is not None:
                add(default_dataset, q, read_text(f), f)
            continue

        try:
            obj = load_jsonish(f)
        except Exception:
            continue

        found = False
        for rec in _iter_report_like(obj):
            qv = first_present(rec, Q_KEYS)
            tv = first_present(rec, REPORT_TEXT_KEYS)
            if qv is None or tv is None:
                continue
            try:
                q = int(str(qv).lstrip("qQ"))
            except Exception:
                continue
            dataset = str(first_present(rec, DATASET_KEYS) or default_dataset)
            add(dataset, q, str(tv), f)
            found = True

        if not found and isinstance(obj, dict):
            for k, v in obj.items():
                try:
                    q = int(str(k).lstrip("qQ"))
                except Exception:
                    continue
                if isinstance(v, str):
                    add(default_dataset, q, v, f)
                elif isinstance(v, dict):
                    tv = first_present(v, REPORT_TEXT_KEYS)
                    if tv is not None:
                        dataset = str(first_present(v, DATASET_KEYS) or default_dataset)
                        add(dataset, q, str(tv), f)

    if not out:
        raise ValueError(f"No reports found in {path}")
    return out


def extract_article_ids_from_obj(obj: Any) -> List[str]:
    ids: List[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, list):
            for z in x:
                walk(z)
        elif isinstance(x, dict):
            for k, v in x.items():
                lk = k.lower()
                if lk in ARTICLE_KEYS or lk in ("article_ids", "retrieved_articles", "sources"):
                    if isinstance(v, str):
                        ids.append(canonical_article_id(v))
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, str):
                                ids.append(canonical_article_id(item))
                            elif isinstance(item, dict):
                                aid = first_present(item, ARTICLE_KEYS)
                                if aid is not None:
                                    ids.append(canonical_article_id(str(aid)))
                                walk(item)
                    elif isinstance(v, dict):
                        aid = first_present(v, ARTICLE_KEYS)
                        if aid is not None:
                            ids.append(canonical_article_id(str(aid)))
                        walk(v)
                else:
                    walk(v)

    walk(obj)
    seen = set()
    out = []
    for aid in ids:
        if aid and aid not in seen:
            seen.add(aid)
            out.append(aid)
    return out


def load_retrieval_map(path: Path, default_dataset: str) -> Dict[Tuple[str, int], List[str]]:
    result: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    files = [p for p in files if p.is_file() and p.suffix.lower() in (".json", ".jsonl", ".ndjson")]

    for f in files:
        try:
            obj = load_jsonish(f)
        except Exception:
            continue

        q_from_name = parse_q_number_from_name(f)

        records: List[Mapping[str, Any]] = []
        def collect(x: Any) -> None:
            if isinstance(x, list):
                for z in x:
                    collect(z)
            elif isinstance(x, dict):
                if any(k in x for k in Q_KEYS):
                    records.append(x)
                else:
                    for v in x.values():
                        collect(v)
        collect(obj)

        if records:
            for rec in records:
                qv = first_present(rec, Q_KEYS)
                try:
                    q = int(str(qv).lstrip("qQ"))
                except Exception:
                    continue
                ds = str(first_present(rec, DATASET_KEYS) or default_dataset)
                result[(ds, q)].extend(extract_article_ids_from_obj(rec))
        elif q_from_name is not None:
            result[(default_dataset, q_from_name)].extend(extract_article_ids_from_obj(obj))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                try:
                    q = int(str(k).lstrip("qQ"))
                except Exception:
                    continue
                result[(default_dataset, q)].extend(extract_article_ids_from_obj(v))

    for k, ids in list(result.items()):
        seen = set()
        result[k] = [x for x in ids if x and not (x in seen or seen.add(x))]
    return dict(result)


def retrieval_from_final_reports(report_path: Path, condition: str, default_dataset: str) -> Dict[Tuple[str, int], List[str]]:
    if report_path.is_dir():
        raise ValueError(f"Expected one final JSON report containing embedded logs, got directory: {report_path}")
    obj = load_jsonish(report_path)
    records = list(_iter_report_like(obj))
    if not records:
        raise ValueError(f"No per-question final reports in {report_path}; settings JSON is not a report.")
    result: Dict[Tuple[str, int], List[str]] = {}
    for rec in records:
        qv = first_present(rec, Q_KEYS)
        dataset = str(first_present(rec, DATASET_KEYS) or default_dataset)
        q = int(qv)
        key = (dataset, q)
        if key in result:
            raise ValueError(f"Duplicate embedded retrieval entry: {condition} {key}")
        if 'retrieval_log' not in rec or not isinstance(rec['retrieval_log'], list):
            raise ValueError(f"Missing/non-list retrieval_log: {condition} {key} in {report_path}")
        if rec.get('condition') not in (None, condition):
            raise ValueError(f"Wrong report condition: expected {condition}, got {rec.get('condition')} for {key}")
        logs = rec['retrieval_log']
        article_ids = []
        for event in logs:
            if not isinstance(event, dict) or not isinstance(event.get('results'), list):
                raise ValueError(f"Malformed retrieval_log event: {condition} {key}")
            for hit in event['results']:
                if isinstance(hit, dict) and hit.get('article_id'):
                    article_ids.append(canonical_article_id(str(hit['article_id'])))
        result[key] = list(dict.fromkeys(article_ids))
    return result


def condition_retrieval_map(condition_cfg: Mapping[str, Any], condition: str, default_dataset: str) -> Dict[Tuple[str, int], List[str]]:
    explicit = condition_cfg.get('retrieval_log')
    if explicit:
        return load_retrieval_map(Path(explicit), default_dataset)
    return retrieval_from_final_reports(Path(condition_cfg['report']), condition, default_dataset)


def numeric_payload(text: str) -> Tuple[str, ...]:
    vals = []
    for m in re.finditer(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", text.replace(",", "")):
        token = m.group(0)
        try:
            d = Decimal(token)
            vals.append(format(d.normalize(), "f"))
        except InvalidOperation:
            vals.append(token)
    return tuple(vals)


def value_changed_after_rounding(original: str, perturbed: str) -> int:
    a = numeric_payload(original)
    b = numeric_payload(perturbed)
    if a and b:
        return int(a != b)
    return int(normalize_ws(original) != normalize_ws(perturbed))


@dataclass
class Reconstruction:
    modified_text: str
    entry_modified_spans: Dict[int, Tuple[int, int]]


def reconstruct_article(original_text: str, entries: Sequence[MutationEntry]) -> Reconstruction:
    sortable = []
    for e in entries:
        try:
            a = int(e.source_start)
            b = int(e.source_end)
        except Exception as ex:
            raise ValueError(f"Non-integer source offsets in {e.article_id} entry {e.entry_index}: {e.source_start},{e.source_end}") from ex
        sortable.append((a, b, e))

    sortable.sort(key=lambda x: (x[0], x[1], x[2].entry_index))
    prev_end = -1
    for a, b, e in sortable:
        if a < prev_end:
            raise ValueError(f"Overlapping mutation spans in {e.article_id}")
        prev_end = b
        if original_text[a:b] != e.original:
            raise ValueError(
                f"Source bytes/span mismatch for {e.article_id} entry {e.entry_index}: "
                f"original[{a}:{b}]={original_text[a:b]!r}, log={e.original!r}"
            )

    chunks = []
    cursor = 0
    out_pos = 0
    spans: Dict[int, Tuple[int, int]] = {}
    for a, b, e in sortable:
        prefix = original_text[cursor:a]
        chunks.append(prefix)
        out_pos += len(prefix)
        ms = out_pos
        chunks.append(e.perturbed)
        out_pos += len(e.perturbed)
        me = out_pos
        spans[e.entry_index] = (ms, me)
        cursor = b
    chunks.append(original_text[cursor:])
    return Reconstruction("".join(chunks), spans)


def build_article_index(root: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for p in root.rglob("*.txt"):
        key = canonical_article_id(p.name)
        if key in idx and idx[key] != p:
            raise ValueError(f"Duplicate article basename under source root: {key}: {idx[key]} vs {p}")
        idx[key] = p
    return idx


_SOURCE_EXPRESSIONS_NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?|[.,]\d+)"
_SOURCE_EXPRESSIONS_NUMERIC_CORE = re.compile(
    _SOURCE_EXPRESSIONS_NUMBER + r"(?:\s*(?:/|:|to|–|—|−|-)\s*" + _SOURCE_EXPRESSIONS_NUMBER + r")*",
    re.I,
)
_SOURCE_EXPRESSIONS_UNIT = re.compile(r"\s*(?:[%°]|[A-Za-zµμ][A-Za-z0-9µμ°%/²³^_-]*)")


def _source_expressions_candidate_expressions(text: str) -> List[Tuple[int, int]]:
    found: List[Tuple[int, int]] = []
    for m in _SOURCE_EXPRESSIONS_NUMERIC_CORE.finditer(text):
        a, b = m.span()
        if a and (text[a-1].isdigit() or text[a-1] in '.,'):
            continue
        unit = _SOURCE_EXPRESSIONS_UNIT.match(text, b)
        if unit:
            raw = unit.group(0)
            token = raw.strip()
            known = {'%', '°', 'HU', 'mm', 'cm', 'm', 'km', 'kg', 'g', 'mg', 'µg',
                     'μg', 'ml', 'mL', 'L', 'mmHg', 'kPa', 'Pa', 'years', 'year',
                     'months', 'month', 'days', 'day', 'weeks', 'week', 'hours',
                     'hour', 'minutes', 'minute', 'min', 's', 'ms', 'Hz', 'MHz',
                     'Gy', 'mGy', 'cmH2O', 'bpm', 'IU', 'mIU', 'mmol/L', 'mg/dL',
                     'm/s', 'cm/s', 'cc', 'degrees'}
            if token in known or token.lower() in {x.lower() for x in known}:
                b = unit.end()
        found.append((a, b))
    return found


def resolve_complete_source_expression(text: str, start: int, end: int) -> Dict[str, Any]:
    if start < 0 or end <= start or end > len(text):
        return {'status': 'unresolved', 'reason': 'invalid_source_offsets',
                'expression': '', 'start': '', 'end': ''}
    choices = [(a,b) for a,b in _source_expressions_candidate_expressions(text) if a <= start and b >= end]
    if len(choices) != 1:
        return {'status': 'unresolved',
                'reason': 'no_unambiguous_complete_expression' if not choices else 'overlapping_expressions',
                'expression': '', 'start': '', 'end': ''}
    a,b = choices[0]
    return {'status':'resolved', 'reason':'', 'expression':text[a:b], 'start':a, 'end':b}


def source_boundary_defect(text: str, start: int, end: int) -> str:
    defects=[]
    if start > 0:
        left = text[start-1]
        if left.isdigit() or (left in ',.' and start > 1 and text[start-2].isdigit()):
            defects.append('left_numeric_token_continuation')
        elif left in '/:–—−-' and start > 1 and text[start-2].isdigit():
            defects.append('left_ratio_or_range_component')
    if end < len(text):
        right=text[end]
        if right.isdigit() or (right in ',.' and end+1<len(text) and text[end+1].isdigit()):
            defects.append('right_numeric_token_continuation')
        elif right in '/:–—−-' and end+1<len(text) and text[end+1].isdigit():
            defects.append('right_ratio_or_range_component')
    return ';'.join(defects)


def source_expressions_source_entry(original_text: str, recon: Reconstruction, entry: MutationEntry) -> Dict[str, Any]:
    try:
        os, oe = int(entry.source_start), int(entry.source_end)
        ms, me = recon.entry_modified_spans[entry.entry_index]
    except (ValueError, TypeError, KeyError):
        return {'status':'unresolved', 'reason':'missing_span_mapping', 'partial_source':0,
                'boundary_defect':'', 'original_expression':'', 'modified_expression':'',
                'original_expression_start':'','original_expression_end':'',
                'modified_expression_start':'','modified_expression_end':''}
    o=resolve_complete_source_expression(original_text, os, oe)
    m=resolve_complete_source_expression(recon.modified_text, ms, me)
    defect=source_boundary_defect(original_text, os, oe)
    status='resolved' if o['status']=='resolved' and m['status']=='resolved' else 'unresolved'
    return {'status':status,
            'reason':';'.join(x for x in (o['reason'],m['reason']) if x),
            'partial_source':int(bool(defect)), 'boundary_defect':defect,
            'original_expression':o['expression'], 'modified_expression':m['expression'],
            'original_expression_start':o['start'], 'original_expression_end':o['end'],
            'modified_expression_start':m['start'], 'modified_expression_end':m['end']}


@dataclass
class InputEvidence:
    status: str
    span_present: str
    text_path: str
    text_sha256: str
    source_start: Any
    source_end: Any


DET_FIELDS = [
    "pair_id",
    "condition",
    "dataset",
    "q_number",
    "article_id",
    "original",
    "perturbed",
    "source_start",
    "source_end",
    "entry_index",
    "value_changed",
    "original_complete_number_match",
    "substring_match",
    "complete_number_match",
    "substring_spans_json",
    "complete_number_spans_json",
    "normalization_rule",
    "report_path",
    "report_sha256",
    "source_article_path",
    "source_article_sha256",
    "modified_source_sha256",
    "original_span_start",
    "original_span_end",
    "modified_span_start",
    "modified_span_end",
    "input_evidence",
    "input_span_present",
    "input_text_path",
    "input_text_sha256",
    "input_source_start",
    "input_source_end",
    "reconstruction_contract_hashes_json",
    "expression_status", "expression_reason", "partial_source",
    "boundary_defect", "original_expression", "modified_expression",
    "original_expression_start", "original_expression_end",
    "modified_expression_start", "modified_expression_end",
]


def deterministic_condition_rows(
    condition: str,
    condition_cfg: Mapping[str, Any],
    default_dataset: str,
    source_article_index: Mapping[str, Path],
    project_root: Path,
) -> List[Dict[str, Any]]:
    reports = load_reports(Path(condition_cfg["report"]), condition, default_dataset)
    retrieval = condition_retrieval_map(condition_cfg, condition, default_dataset)

    mutation_log_path = Path(condition_cfg["mutation_log"])
    mutations = load_mutation_log(mutation_log_path)


    reconstructed_cache: Dict[str, Reconstruction] = {}
    source_cache: Dict[str, Tuple[str, str, Path]] = {}
    source_expressions_expression_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}

    rows: List[Dict[str, Any]] = []

    for (dataset, q), rep in sorted(reports.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        retrieved = retrieval.get((dataset, q), [])
        seen = set()
        distinct = [x for x in retrieved if not (x in seen or seen.add(x))]

        for rid in distinct:
            aid = canonical_article_id(rid)
            entries = mutations.get(aid, [])
            if not entries:
                continue

            source_path = source_article_index.get(aid)
            source_text = ""
            source_hash = ""
            if source_path is not None:
                if aid not in source_cache:
                    txt = read_text(source_path)
                    source_cache[aid] = (txt, file_sha256(source_path), source_path)
                source_text, source_hash, source_path = source_cache[aid]

            if source_text and aid not in reconstructed_cache:
                reconstructed_cache[aid] = reconstruct_article(source_text, entries)

            recon = reconstructed_cache.get(aid)

            for e in entries:
                sub_hit, sub_spans, _, _ = substring_spans(rep.report_text, e.perturbed)
                cache_key=(aid,e.entry_index)
                if cache_key not in source_expressions_expression_cache:
                    if recon and source_text:
                        source_expressions_expression_cache[cache_key] = source_expressions_source_entry(source_text, recon, e)
                    else:
                        source_expressions_expression_cache[cache_key] = {
                            'status':'unresolved', 'reason':'source_or_reconstruction_unavailable',
                            'partial_source':0, 'boundary_defect':'',
                            'original_expression':'', 'modified_expression':'',
                            'original_expression_start':'', 'original_expression_end':'',
                            'modified_expression_start':'', 'modified_expression_end':''}
                source_expressions=source_expressions_expression_cache[cache_key]
                if source_expressions['status'] == 'resolved':
                    complete_hit, complete_spans_list, _, _ = complete_number_spans(
                        rep.report_text, source_expressions['modified_expression'])
                    original_complete_hit, original_complete_spans, _, _ = complete_number_spans(
                        rep.report_text, source_expressions['original_expression'])
                else:
                    complete_hit, complete_spans_list = False, []
                    original_complete_hit, original_complete_spans = False, []

                pid = stable_pair_id(
                    condition, dataset, q, e.article_id,
                    e.original, e.perturbed, e.source_start, e.source_end, e.entry_index
                )

                modified_start = ""
                modified_end = ""
                if recon and e.entry_index in recon.entry_modified_spans:
                    modified_start, modified_end = recon.entry_modified_spans[e.entry_index]

                iev = InputEvidence(
                    status="unknown",
                    span_present="unknown",
                    text_path="",
                    text_sha256="",
                    source_start="",
                    source_end="",
                )

                rows.append({
                    "pair_id": pid,
                    "condition": condition,
                    "dataset": dataset,
                    "q_number": q,
                    "article_id": e.article_id,
                    "original": e.original,
                    "perturbed": e.perturbed,
                    "source_start": e.source_start,
                    "source_end": e.source_end,
                    "entry_index": e.entry_index,
                    "value_changed": value_changed_after_rounding(e.original, e.perturbed),
                    "substring_match": int(sub_hit),
                    "complete_number_match": int(complete_hit),
                    "original_complete_number_match": int(original_complete_hit),
                    "original_complete_spans_json": json.dumps(original_complete_spans, ensure_ascii=False),
                    "substring_spans_json": json.dumps(sub_spans, ensure_ascii=False),
                    "complete_number_spans_json": json.dumps(complete_spans_list, ensure_ascii=False),
                    "normalization_rule": r"collapse every \s+ run to one ASCII space; strip leading/trailing whitespace",
                    "report_path": rep.report_path,
                    "report_sha256": rep.report_sha256,
                    "source_article_path": str(source_path) if source_path else "",
                    "source_article_sha256": source_hash,
                    "modified_source_sha256": sha256_text(recon.modified_text) if recon else "",
                    "original_span_start": e.source_start,
                    "original_span_end": e.source_end,
                    "modified_span_start": modified_start,
                    "modified_span_end": modified_end,
                    "input_evidence": iev.status,
                    "input_span_present": iev.span_present,
                    "input_text_path": iev.text_path,
                    "input_text_sha256": iev.text_sha256,
                    "input_source_start": iev.source_start,
                    "input_source_end": iev.source_end,
                    "reconstruction_contract_hashes_json": "",
                    "expression_status": source_expressions['status'],
                    "expression_reason": source_expressions['reason'],
                    "partial_source": source_expressions['partial_source'],
                    "boundary_defect": source_expressions['boundary_defect'],
                    "original_expression": source_expressions['original_expression'],
                    "modified_expression": source_expressions['modified_expression'],
                    "original_expression_start": source_expressions['original_expression_start'],
                    "original_expression_end": source_expressions['original_expression_end'],
                    "modified_expression_start": source_expressions['modified_expression_start'],
                    "modified_expression_end": source_expressions['modified_expression_end'],
                })

    return rows


def resolve_path(base: Path, x: str) -> Path:
    p = Path(x)
    return p if p.is_absolute() else base / p


def absolutize_config_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    project_root = Path(config["project"]["repo_root"]).resolve()
    config["project_root"] = str(project_root)
    numerical_matching = config["numerical_matching"]

    path_keys = (
        "source_article_root",
    )
    for k in path_keys:
        if k in numerical_matching:
            numerical_matching[k] = str(resolve_path(project_root, numerical_matching[k]))

    for key, value in numerical_matching.get("outputs", {}).items():
        if value:
            numerical_matching["outputs"][key] = str(resolve_path(project_root, value))

    for section in ("deterministic_conditions",):
        for cond, cc in numerical_matching.get(section, {}).items():
            for k in ("report", "retrieval_log", "mutation_log", "input_log"):
                if cc.get(k):
                    cc[k] = str(resolve_path(project_root, cc[k]))
    return config


def validate_numerical_matching_config(config: Mapping[str, Any]) -> None:
    if "numerical_matching" not in config:
        raise KeyError('Missing config["numerical_matching"]')
    numerical_matching = config["numerical_matching"]

    det = numerical_matching.get("deterministic_conditions", {})
    if len(det) != 13:
        raise ValueError(
            f'numerical matching requires exactly 13 deterministic report conditions; config currently has {len(det)}.'
        )

    for section_name, section in (("deterministic_conditions", det),):
        for cond, cc in section.items():
            required = ("report", "mutation_log")
            miss = [k for k in required if not cc.get(k)]
            if miss:
                raise KeyError(f"{section_name}.{cond} missing: {miss}")


def run_deterministic(config: Dict[str, Any]) -> Dict[str, Any]:
    validate_numerical_matching_config(config)

    project_root = Path(config["project_root"])
    numerical_matching = config["numerical_matching"]
    output_paths = numerical_matching["outputs"]

    default_dataset = str(numerical_matching.get("dataset", "RadSaFE-200"))
    source_root = Path(numerical_matching["source_article_root"])
    source_index = build_article_index(source_root)

    all_rows: List[Dict[str, Any]] = []

    for cond, cc in numerical_matching["deterministic_conditions"].items():
        rows = deterministic_condition_rows(
            condition=cond,
            condition_cfg=cc,
            default_dataset=default_dataset,
            source_article_index=source_index,
            project_root=project_root,
        )
        all_rows.extend(rows)

    out_csv = Path(output_paths["deterministic_csv"])
    write_csv(out_csv, all_rows, DET_FIELDS)

    inventory_fields = [
        "pair_id", "condition", "report_model", "dataset", "q_number",
        "article_id", "audit_index", "start_char", "end_char", "original",
        "perturbed", "report_sha256", "source_sha256", "modified_source_sha256",
        "value_changed", "substring_match", "complete_number_match",
        "original_complete_number_match", "match_spans_json",
        "input_evidence_status", "changed_span_supplied", "input_text_path",
        "input_text_sha256",
    ]
    model_map = {k: v for k, v in numerical_matching["report_models"].items() if k != "main"}
    default_report_model = numerical_matching["report_models"]["main"]
    inventory_rows = []
    for r in all_rows:
        cond = str(r["condition"])
        model = model_map.get(cond.split("_")[0], default_report_model)
        spans = []
        for kind, spans_field in (("perturbed", "complete_number_spans_json"), ("original", "original_complete_spans_json")):
            for span in json.loads(r[spans_field]):
                spans.append({
                    "version": kind,
                    "start": span["normalized_start"],
                    "end": span["normalized_end"],
                })
        inventory_rows.append({
            "pair_id": r["pair_id"], "condition": cond,
            "report_model": model,
            "dataset": r["dataset"], "q_number": r["q_number"],
            "article_id": r["article_id"], "audit_index": r["entry_index"],
            "start_char": r["source_start"], "end_char": r["source_end"],
            "original": r["original"], "perturbed": r["perturbed"],
            "report_sha256": r["report_sha256"],
            "source_sha256": r["source_article_sha256"],
            "modified_source_sha256": r["modified_source_sha256"],
            "value_changed": r["value_changed"],
            "substring_match": r["substring_match"],
            "complete_number_match": r["complete_number_match"],
            "original_complete_number_match": r["original_complete_number_match"],
            "match_spans_json": json.dumps(spans, ensure_ascii=False),
            "input_evidence_status": ("unavailable" if r["input_evidence"] == "unknown" else r["input_evidence"]),
            "changed_span_supplied": ({"1": "yes", "0": "no"}.get(str(r["input_span_present"]), "unknown")),
            "input_text_path": r["input_text_path"],
            "input_text_sha256": r["input_text_sha256"],
        })
    write_csv(Path(output_paths["match_inventory_csv"]), inventory_rows, inventory_fields)
    return Path(output_paths["match_inventory_csv"])


def run_numerical_matching(config: Dict[str, Any]) -> Path:
    config = absolutize_config_paths(config)
    validate_numerical_matching_config(config)
    return run_deterministic(config)


def main_numerical_matching(config_path):
    return run_numerical_matching(load_config(config_path))
