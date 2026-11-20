"""
src/reported_counts.py
Created on November 18, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import json
import os
from pathlib import Path

import pandas as pd

from src.analyze_numerical_matching import load_reports, normalize_ws_with_map
from src.config import load_config

LEVELS = ["num010", "num025", "num050", "num100"]
RATIO_SEPARATORS = "/:"


def write_table(df, out, name):
    tmp = out / f"{name}.{os.getpid()}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, out / name)


def answer_counts(runs):
    return [dict(quantity="answers", scope="all conditions", value=len(runs)),
            dict(quantity="unparsed_answers", scope="all conditions", value=int((runs.answer_status == "unparsed").sum()))]


def answer_fates(runs):
    models = sorted(runs.model.unique())
    ev = runs[runs.model == models[0]].set_index(["dataset", "q_number", "condition"]).evidence_sha256
    keys = sorted({(d, int(q)) for d, q in zip(runs.dataset, runs.q_number)})
    changed_q = {k for k in keys if ev[(k[0], k[1], "ev_clean")] != ev[(k[0], k[1], "ev_numeric")]}
    subsets = {"ev_numeric_changed": changed_q, "ev_numeric_unchanged": set(keys) - changed_q}
    correct = runs.set_index(["model", "dataset", "q_number", "condition"]).correct
    rows = []
    for subset, compared, reference in [("ev_numeric_changed", "ev_numeric", "ev_clean"), ("ev_numeric_unchanged", "ev_numeric", "ev_clean"),
                                        ("ev_numeric_changed", "num100", "clean")]:
        lost = gained = n = 0
        for model in models:
            for d, q in sorted(subsets[subset]):
                a, b = int(correct[(model, d, q, reference)]), int(correct[(model, d, q, compared)])
                lost += int(a == 1 and b == 0)
                gained += int(a == 0 and b == 1)
                n += 1
        scope = f"{compared}_minus_{reference}|{subset}"
        rows += [dict(quantity="questions", scope=scope, value=len(subsets[subset])), dict(quantity="answers", scope=scope, value=n),
                 dict(quantity="answers_correct_to_incorrect", scope=scope, value=lost),
                 dict(quantity="answers_incorrect_to_correct", scope=scope, value=gained)]
    return rows


def partial_matches(config):
    clean = Path(config["corpora"]["clean"])
    texts = {}
    rows = []
    for level in LEVELS:
        log = json.loads(Path(config["numerical_matching"]["deterministic_conditions"][level]["mutation_log"]).read_text(encoding="utf-8"))
        after_separator = before_digit = after_digit = partial = 0
        articles = set()
        for e in log:
            if e["article_id"] not in texts:
                texts[e["article_id"]] = (clean / e["article_id"]).read_text(encoding="utf-8", errors="ignore")
            t, s, en = texts[e["article_id"]], int(e["start_char"]), int(e["end_char"])
            if t[s:en] != e["original"]:
                raise ValueError(f"{level} {e['article_id']}: the logged span differs from the clean article")
            a = s > 1 and t[s - 1] == "," and t[s - 2].isdigit()
            b = en < len(t) and t[en].isdigit()
            c = s > 0 and t[s - 1].isdigit()
            after_separator += int(a)
            before_digit += int(b)
            after_digit += int(c)
            if a or b or c:
                partial += 1
                articles.add(e["article_id"])
        rows += [dict(quantity="logged_changes", scope=level, value=len(log)),
                 dict(quantity="partial_matches", scope=level, value=partial),
                 dict(quantity="partial_matches_after_thousands_separator", scope=level, value=after_separator),
                 dict(quantity="partial_matches_before_digit", scope=level, value=before_digit),
                 dict(quantity="partial_matches_after_digit", scope=level, value=after_digit),
                 dict(quantity="articles_with_partial_matches", scope=level, value=len(articles))]
    return rows


def inside_ratio(text, start, end):
    left = start > 1 and text[start - 1] in RATIO_SEPARATORS and (text[start - 2].isdigit() or text[start - 2] == "%")
    right = end + 1 < len(text) and text[end] in RATIO_SEPARATORS and text[end + 1].isdigit()
    return left or right


def complete_matches(config):
    nm = config["numerical_matching"]
    inv = pd.read_csv(nm["outputs"]["match_inventory_csv"], encoding="utf-8-sig", keep_default_na=False,
                      usecols=["condition", "dataset", "q_number", "value_changed", "complete_number_match",
                               "original_complete_number_match", "match_spans_json"])
    texts = {}
    for condition, cc in nm["deterministic_conditions"].items():
        for (dataset, q), rep in load_reports(Path(cc["report"]), condition, str(nm.get("dataset", "RadSaFE-200"))).items():
            texts[(condition, dataset, q)] = normalize_ws_with_map(rep.report_text).text
    matches = in_ratio = 0
    for r in inv[inv.value_changed.astype(int) == 1].itertuples():
        spans = json.loads(r.match_spans_json)
        t = texts[(r.condition, str(r.dataset), int(r.q_number))]
        for version, flag in (("perturbed", r.complete_number_match), ("original", r.original_complete_number_match)):
            if int(flag) != 1:
                continue
            matches += 1
            sp = [s for s in spans if s["version"] == version]
            if sp and all(inside_ratio(t, s["start"], s["end"]) for s in sp):
                in_ratio += 1
    scope = f"{inv.condition.nunique()} runs"
    return [dict(quantity="matching_candidates", scope=scope, value=len(inv)),
            dict(quantity="complete_matches_of_changed_values", scope=scope, value=matches),
            dict(quantity="complete_matches_inside_ratio", scope=scope, value=in_ratio)]


def run_reported_counts(config):
    out = Path(config["reported_counts"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(Path(config["result_tables"]["output_dir"]) / "runs.csv", keep_default_na=False)
    rows = answer_counts(runs) + answer_fates(runs) + partial_matches(config) + complete_matches(config)
    write_table(pd.DataFrame(rows), out, "reported_counts.csv")
    return out


def main_reported_counts(config_path):
    return run_reported_counts(load_config(config_path))
