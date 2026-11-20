"""
src/result_tables.py
Created on November 17, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import hashlib
import json
import os
from decimal import ROUND_DOWN, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from Inference.report_utils import add_fdr, report_metric, report_paired_diff, set_scale
from src.config import load_config

ORDER = ["RadioRAG", "RaR", "New-31"]
LEVEL_METRICS = ["accuracy", "high_risk_error_rate", "unsafe_error_rate", "contradiction_error_rate",
                 "changed_answer_rate", "mean_confidence"]
LABEL_METRICS = {"high_risk_error_rate": "high_risk", "unsafe_error_rate": "unsafe",
                 "contradiction_error_rate": "contradicts"}
PERF = ["mean", "std", "ci_low", "ci_high", "mean_raw", "std_raw", "ci_low_raw", "ci_high_raw", "point_raw", "central_estimator"]
REPORT_VARIANTS = ("dsv4_", "q35_")
GROUPS = [
    ("vs_clean", [(c, "clean") for c in ["num010", "num025", "num050", "num100", "cov010", "cov025",
                                         "cov050", "cov100", "noretrieval", "clean_r2", "clean_r3"]]),
    ("volume", [(c, "topk05_pad00") for c in ["topk01_pad00", "topk03_pad00", "topk10_pad00", "topk20_pad00",
                                              "topk05_pad50", "topk05_pad75", "topk05_pad90", "topk05_pad95"]]),
    ("vs_closedbook", [(c, "closedbook") for c in ["clean", "noretrieval", "ev_clean"]]),
    ("vs_ev_clean", [(c, "ev_clean") for c in ["ev_conflict", "ev_numeric", "ev_forum", "ev_overconfident",
                                               "ev_hedged", "ev_review"]]),
    ("top1", [("top1_100", "clean")]),
    ("report_models", [(p + s, p + "clean") for p in REPORT_VARIANTS
                       for s in ["clean_r2", "num100", "cov100", "noretrieval"]]
     + [("dsv4_clean", "clean"), ("q35_clean", "clean")]),
]
DIRECT = ["rag_clean", "rag_num100"]
DIRECT_CONTRASTS = {
    "rag_num100_minus_rag_clean": ("rag_num100", "rag_clean"),
    "rag_clean_minus_closedbook": ("rag_clean", "closedbook"),
    "rag_clean_minus_clean": ("rag_clean", "clean"),
    "rag_num100_minus_num100": ("rag_num100", "num100"),
    "rag_change_minus_report_change": (("rag_num100", "rag_clean"), ("num100", "clean")),
}
INPUTS = {}


def used(path):
    INPUTS[str(Path(path))] = None
    return path


def family_level(condition):
    stem = condition[len("dsv4_"):] if condition.startswith("dsv4_") else condition
    stem = stem[len("q35_"):] if stem.startswith("q35_") else stem
    if stem in ("clean", "clean_r2", "clean_r3"):
        return "clean", ""
    if stem in ("closedbook", "noretrieval"):
        return stem, ""
    if stem.startswith("num"):
        return "numeric", str(int(stem[3:]))
    if stem.startswith("cov"):
        return "coverage", str(int(stem[3:]))
    if stem in ("ev_clean", "ev_conflict", "ev_numeric"):
        return "evidence", ""
    if stem.startswith("ev_"):
        return "style", ""
    if stem == "rag_clean":
        return "direct_retrieval", "0"
    if stem == "rag_num100":
        return "direct_retrieval", "100"
    if stem == "top1_100":
        return "top1_removal", "100"
    if stem.startswith("topk"):
        k, pad = int(stem[4:6]), int(stem[-2:])
        return ("volume_padding", str(pad)) if pad else ("volume_count", str(k))
    raise ValueError(f"Unknown condition: {condition}")


def reference(condition):
    if condition in ("clean", "ev_clean"):
        return None
    if condition.startswith("topk"):
        return "topk05_pad00"
    if condition.startswith("ev_"):
        return "ev_clean"
    if condition == "rag_clean":
        return "closedbook"
    if condition == "rag_num100":
        return "rag_clean"
    for prefix in REPORT_VARIANTS:
        if condition.startswith(prefix) and condition != prefix + "clean":
            return prefix + "clean"
    return "clean"


def answer_path(cfg, model, condition):
    return Path(cfg["answers_root"]) / model.replace("/", "_") / f"{condition}.json"


def report_path(config, condition):
    if condition.startswith(REPORT_VARIANTS):
        return Path(config["outputs"]["report_models_source_ablation"]) / "reports" / f"{condition}.json"
    return Path(config["outputs"]["formal_reports"]) / f"{condition}.json"


def trunc16(df):
    for col in df.columns:
        if df[col].dtype.kind == "f":
            df[col] = [float(Decimal(repr(v)).quantize(Decimal("1e-16"), rounding=ROUND_DOWN)) if pd.notna(v) else v
                       for v in df[col]]
    return df


def write_table(df, out, name):
    tmp = out / f"{name}.{os.getpid()}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, out / name)


def packed(res, unit):
    row = {k: res[f"x_{k}"] for k in PERF}
    row["n"] = res["n"]
    row["unit"] = unit
    if "p_raw" in res:
        row["p_raw"] = res["p_raw"]
        row["test_method"] = res["test_method"]
    return row


def in_interval(row):
    return bool(row["ci_low_raw"] <= row["mean_raw"] <= row["ci_high_raw"])


def fdr(rows, family_col="fdr_family"):
    df = pd.DataFrame(rows)
    return add_fdr(df, family_cols=[family_col])


def load_benchmark(path):
    bench = pd.read_csv(used(path), encoding="utf-8-sig", low_memory=False)
    keys = sorted(((r.dataset, int(r.q_number)) for r in bench.itertuples()),
                  key=lambda k: (ORDER.index(k[0]), k[1]))
    if len(keys) != 200 or len(set(keys)) != len(keys):
        raise ValueError("The benchmark must contain 200 unique question keys.")
    return bench.set_index(["dataset", "q_number"]), keys


def validate_question_keys(records, expected, label):
    actual = [(str(r["dataset"]), int(r["q_number"])) for r in records]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError(f"{label}: duplicate, missing, or unexpected question keys")


def benchmark_records(source_csv):
    frame = pd.read_csv(source_csv, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    records = []
    for i, row in frame.iterrows():
        record = dict(question_id=i, q_number=int(row.q_number), dataset=row.dataset.strip(),
                      summary=row.question.strip(),
                      options={chr(64 + j): str(row[f"option_{j}"]).strip() for j in range(1, 6)
                               if str(row[f"option_{j}"]).strip()}, answer_index=int(row.answer_index))
        for field in ["question_type", "subspecialty", "source_url", "source_case_title"]:
            record[field] = str(row[field]).strip() or None
        if chr(64 + record["answer_index"]) not in record["options"]:
            raise ValueError(f"Gold answer outside available options: {record['dataset']} {record['q_number']}")
        records.append(record)
    validate_question_keys(records, [(r["dataset"], r["q_number"]) for r in records], "Benchmark CSV")
    return records


def main_benchmark(config_path):
    config = load_config(config_path)
    records = benchmark_records(config["benchmark"]["source_csv"])
    target = Path(config["benchmark"]["runtime_json"])
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != records:
            raise ValueError("Existing runtime JSON differs from the benchmark CSV.")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(f".{os.getpid()}.tmp")
    temp.write_bytes(json.dumps(records, ensure_ascii=False, indent=2).replace("\n", "\r\n").encode("utf-8"))
    os.replace(temp, target)
    return target


def option_labels(bench, key, letter):
    i = "ABCDE".index(letter) + 1
    row = bench.loc[key]
    return {lab: str(int(row[f"option_{i}_{lab}"])) for lab in ("high_risk", "unsafe", "contradicts")}


def build_runs(config, cfg, bench, models, groups):
    rows = []
    for group in groups:
        for model in sorted(models):
            for condition in sorted(group):
                answers = json.loads(used(answer_path(cfg, model, condition)).read_text(encoding="utf-8"))
                if len(answers) != 200:
                    raise ValueError(f"{model} {condition}: expected 200 answers, got {len(answers)}")
                validate_question_keys(answers, bench.index, f"{model} {condition}")
                reports = {}
                family, level = family_level(condition)
                rp = report_path(config, condition)
                if family not in ("closedbook", "evidence", "style", "direct_retrieval"):
                    for r in json.loads(used(rp).read_text(encoding="utf-8")):
                        reports[(str(r["dataset"]), int(r["q_number"]))] = r
                for a in sorted(answers, key=lambda r: (str(r["dataset"]), int(r["q_number"]))):
                    key = (str(a["dataset"]), int(a["q_number"]))
                    letter = a.get("answer_letter") or ""
                    gold = "ABCDE"[int(bench.loc[key]["answer_index"]) - 1]
                    labels = option_labels(bench, key, letter) if letter else {"high_risk": "", "unsafe": "", "contradicts": ""}
                    rep = reports.get(key, {})
                    conf = a.get("confidence")
                    rows.append(dict(model=model, condition=condition, report_model=rep.get("report_model_requested", ""),
                                     family=family, level=level, dataset=key[0], q_number=key[1],
                                     evidence_sha256=a.get("evidence_sha256") or "", report_status=rep.get("report_status", ""),
                                     answer_letter=letter, answer_status=a.get("answer_status") or "",
                                     correct=int(letter != "" and letter == gold),
                                     confidence="" if conf is None else float(conf), finish_reason=a.get("finish_reason") or "",
                                     chosen_high_risk=labels["high_risk"], chosen_unsafe=labels["unsafe"],
                                     chosen_contradicts=labels["contradicts"]))
    return pd.DataFrame(rows)


def answer_arrays(runs, keys, second):
    index = {k: i for i, k in enumerate(keys)}
    out = {}
    for (model, condition), g in runs.groupby(["model", "condition"]):
        validate_question_keys(g.to_dict("records"), keys, f"{model} {condition}")
        a = dict(letter=np.array([""] * len(keys), dtype=object), correct=np.zeros(len(keys)),
                 parsed=np.zeros(len(keys)), conf=np.full(len(keys), np.nan),
                 high_risk=np.zeros(len(keys)), unsafe=np.zeros(len(keys)), contradicts=np.zeros(len(keys)),
                 second_high_risk=np.zeros(len(keys)))
        for r in g.itertuples():
            i = index[(r.dataset, int(r.q_number))]
            a["letter"][i] = r.answer_letter
            a["parsed"][i] = float(r.answer_letter != "")
            a["correct"][i] = float(r.correct)
            a["conf"][i] = float(r.confidence) / 100 if r.confidence != "" else np.nan
            wrong = int(r.correct) == 0
            for lab in LABEL_METRICS.values():
                a[lab][i] = float(wrong and str(getattr(r, "chosen_" + lab)) == "1")
            if wrong and r.answer_letter:
                a["second_high_risk"][i] = float(second[(r.dataset, int(r.q_number), r.answer_letter, "high_risk")] == 1)
        out[(model, condition)] = a
    return out


def values_of(a, metric, prefix=""):
    if metric == "accuracy":
        return a["correct"]
    return a[prefix + LABEL_METRICS[metric]]


def changed(a, b):
    return ((a["parsed"] == 1) & (b["parsed"] == 1) & (a["letter"] != b["letter"])).astype(float)


def level_rows(ans, models, conditions):
    rows = []
    for model in models:
        for condition in conditions:
            a = ans[(model, condition)]
            ref = reference(condition)
            for metric in LEVEL_METRICS:
                if metric == "changed_answer_rate":
                    if ref is None:
                        continue
                    res = report_metric(changed(a, ans[(model, ref)]), "x")
                elif metric == "mean_confidence":
                    conf = a["conf"]
                    res = report_metric(conf[np.isfinite(conf)], "x")
                else:
                    res = report_metric(values_of(a, metric), "x")
                rows.append(dict(model=model, condition=condition, subset="all", metric=metric,
                                 **packed(res, "percent")))
    return rows


def contrast_rows(ans, models, metrics, prefix="", extra=None):
    rows = []
    for family, pairs in GROUPS:
        for a_cond, b_cond in pairs:
            for model in models:
                for metric in metrics:
                    res = report_paired_diff(values_of(ans[(model, a_cond)], metric, prefix),
                                             values_of(ans[(model, b_cond)], metric, prefix), "x")
                    rows.append(dict(model=model, subset="all", metric=metric, contrast=f"{a_cond}_minus_{b_cond}",
                                     condition_a=a_cond, condition_b=b_cond, **(extra or {}),
                                     **packed(res, "percentage_points"), fdr_family=f"{family}|{metric}"))
    return rows


def changed_contrast_rows(ans, models, conditions):
    rows = []
    for repeat in ["clean_r2", "clean_r3"]:
        for condition in conditions:
            ref = reference(condition)
            if ref is None or condition == repeat or condition.startswith("rag_") or condition.endswith("clean_r2") and condition.startswith(REPORT_VARIANTS):
                continue
            prefix = next((p for p in REPORT_VARIANTS if ref == p + "clean"), "")
            if prefix and repeat == "clean_r3":
                continue
            rep, rep_ref = (prefix + "clean_r2", prefix + "clean") if prefix else (repeat, "clean")
            for model in models:
                d_a = changed(ans[(model, condition)], ans[(model, ref)])
                d_b = changed(ans[(model, rep)], ans[(model, rep_ref)])
                res = report_paired_diff(d_a, d_b, "x")
                rows.append(dict(model=model, subset="all", metric="changed_answer_rate",
                                 contrast=f"change_{condition}_vs_{ref}_minus_change_{rep}_vs_{rep_ref}",
                                 condition_a=condition, condition_b=ref, **packed(res, "percentage_points"),
                                 fdr_family=f"changed_minus_{repeat}|changed_answer_rate"))
    return rows


def direct_contrast_rows(ans, models):
    contrasts = []
    for model in models:
        for metric in ["accuracy", "high_risk_error_rate"]:
            for name, (pa, pb) in DIRECT_CONTRASTS.items():
                if isinstance(pa, tuple):
                    va = values_of(ans[(model, pa[0])], metric) - values_of(ans[(model, pa[1])], metric)
                    vb = values_of(ans[(model, pb[0])], metric) - values_of(ans[(model, pb[1])], metric)
                else:
                    va, vb = values_of(ans[(model, pa)], metric), values_of(ans[(model, pb)], metric)
                res = report_paired_diff(va, vb, "x")
                contrasts.append(dict(model=model, contrast=name, metric=metric, **packed(res, "percentage_points"),
                                      fdr_family=f"direct_retrieval_{metric}"))
    return contrasts


def second_reader_labels(path):
    sec = pd.read_csv(used(path), encoding="utf-8-sig")
    out = {}
    for r in sec.itertuples():
        for i, letter in enumerate("ABCDE", start=1):
            v = getattr(r, f"option_{i}_high_risk")
            if pd.notna(v):
                out[(r.dataset, int(r.q_number), letter, "high_risk")] = int(v)
    return out


def run_result_tables(config):
    set_scale("percent")
    INPUTS.clear()
    cfg = config["result_tables"]
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    bench, keys = load_benchmark(config["benchmark"]["source_csv"])
    runtime = json.loads(used(Path(config["benchmark"]["runtime_json"])).read_text(encoding="utf-8"))
    if runtime != benchmark_records(config["benchmark"]["source_csv"]):
        raise ValueError("Runtime question text, options, or labels differ from the benchmark CSV.")
    models = sorted(config["models"]["answering"])
    conditions = sorted(c for group in config["conditions"].values() for c in group) + DIRECT

    runs = build_runs(config, cfg, bench, models, list(config["conditions"].values()) + [DIRECT])
    write_table(runs, out, "runs.csv")
    runs = pd.read_csv(out / "runs.csv", keep_default_na=False)
    second = second_reader_labels(cfg["second_reader_csv"])
    ans = answer_arrays(runs, keys, second)
    conditions = sorted(conditions)

    metrics = pd.DataFrame(level_rows(ans, models, conditions))
    metrics["mean_in_interval"] = metrics.apply(in_interval, axis=1)
    write_table(trunc16(metrics), out, "metrics.csv")

    rows = contrast_rows(ans, models, ["accuracy", "high_risk_error_rate"]) + changed_contrast_rows(ans, models, conditions)
    contrasts = fdr(rows)
    contrasts["mean_in_interval"] = contrasts.apply(in_interval, axis=1)
    cols = ["model", "subset", "metric", "contrast", "condition_a", "condition_b"] + PERF + ["n", "unit", "p_raw", "test_method",
                                                                                          "fdr_family", "p_fdr", "significant_fdr05", "mean_in_interval"]
    write_table(trunc16(contrasts[cols]), out, "contrasts.csv")

    dc = fdr(direct_contrast_rows(ans, models))
    dc["mean_in_interval"] = dc.apply(in_interval, axis=1)
    cols = ["model", "contrast", "metric"] + PERF + ["n", "unit", "p_raw", "test_method", "fdr_family", "p_fdr", "significant_fdr05",
                                                     "mean_in_interval"]
    write_table(trunc16(dc[cols]), out, "retrieval_contrasts.csv")

    rows = []
    for model in models:
        for condition in conditions:
            res = report_metric(values_of(ans[(model, condition)], "high_risk_error_rate", "second_"), "x")
            rows.append(dict(model=model, condition=condition, subset="all", metric="high_risk_error_rate",
                             labels="second_reader", **packed(res, "percent")))
    write_table(pd.DataFrame(rows), out, "metrics_second_reader.csv")
    rows = contrast_rows(ans, models, ["high_risk_error_rate"], "second_", {"labels": "second_reader"})
    c2 = fdr(rows)
    cols = ["model", "subset", "metric", "contrast", "condition_a", "condition_b", "labels"] + PERF + [
        "n", "unit", "p_raw", "test_method", "fdr_family", "p_fdr", "significant_fdr05"]
    write_table(c2[cols], out, "contrasts_second_reader.csv")
    for p in INPUTS:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        INPUTS[p] = h.hexdigest()
    params = dict(module="src/result_tables.py", module_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
                  n_boot=1000, boot_seed=0, scale="percent", central_estimator="bootstrap_mean",
                  binary_test="exact_mcnemar", inputs=INPUTS,
                  inference_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (Path(__file__).resolve().parents[1] / "Inference").glob("*.py")})
    tmp = out / f"build_params_result_tables.{os.getpid()}.tmp"
    json.dump(params, open(tmp, "w"), indent=1)
    os.replace(tmp, out / "build_params_result_tables.json")
    return out


def main_result_tables(config_path):
    return run_result_tables(load_config(config_path))
