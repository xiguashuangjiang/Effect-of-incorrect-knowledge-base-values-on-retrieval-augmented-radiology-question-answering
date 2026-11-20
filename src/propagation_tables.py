"""
src/propagation_tables.py
Created on November 17, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from Inference.report_utils import add_fdr, report_metric, report_paired_diff, set_scale
from src.config import load_config
from src.result_tables import validate_question_keys
from src.analyze_numerical_matching import MutationEntry, reconstruct_article, source_expressions_source_entry, value_changed_after_rounding

MAIN, DSV4, Q35 = "nvidia/Qwen3.8-Flash-Next-NVFP4", "MJPansa/DeepSeek-V4-Flash-0731-NVFP4", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
GEN = {"main": dict(model=MAIN, pre="", clean=["clean", "clean_r2", "clean_r3"], changed={"010": "num010", "025": "num025", "050": "num050", "100": "num100"}),
       "dsv4": dict(model=DSV4, pre="dsv4_", clean=["dsv4_clean", "dsv4_clean_r2"], changed={"100": "dsv4_num100"}),
       "q35": dict(model=Q35, pre="q35_", clean=["q35_clean", "q35_clean_r2"], changed={"100": "q35_num100"})}
PERF = ["mean", "std", "ci_low", "ci_high", "mean_raw", "std_raw", "ci_low_raw", "ci_high_raw", "point_raw", "central_estimator"]
INPUTS = {}
UNIT_CLASSES = [("percent", r"%"), ("length", r"\b(mm|cm|m|um)\b"), ("attenuation", r"\bhu\b"),
                ("time or age", r"\b(years?|yrs?|months?|weeks?|days?|hours?|hrs?|minutes?|mins?|seconds?|secs?|ms|msec)\b"),
                ("volume", r"\b(ml|cc|l|litres?|liters?)\b"), ("pressure", r"\bmmhg\b"), ("angle", r"°|\bdegrees?\b"),
                ("dose or acquisition", r"\b(msv|sv|gy|mgy|kv|kvp|kev|mev|mas|t|tesla|mhz|khz|hz|mbq)\b"),
                ("mass or concentration", r"\b(g|mg|kg|ng|ng/ml|mg/dl|g/dl|mmol/l)\b")]
_REP = {}
WS = re.compile(r"\s+")
_LAY = {}
_PAIRS = {}
_ART = {}
_AUG = {}
_PREFIX_RECORDS = {}


def used(path):
    INPUTS[str(Path(path))] = None
    return path


def unpack(d, unit, prefix="x"):
    o = {k: d[f"{prefix}_{k}"] for k in PERF}
    o["n"] = d["n"]
    o["unit"] = unit
    if "p_raw" in d:
        o["p_raw"] = d["p_raw"]
        o["test_method"] = d["test_method"]
    return o


def metric(values, cluster=None):
    return unpack(report_metric(np.asarray(values, float), "x", cluster_ids=cluster), "percent")


def pdiff(a, b):
    return unpack(report_paired_diff(np.asarray(a, float), np.asarray(b, float), "x"), "percentage_points")


def write(df, name):
    tmp = OUT / f"{name}.{os.getpid()}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, OUT / name)


def fdr(rows):
    df = pd.DataFrame(rows)
    has = df["p_raw"].notna() if "p_raw" in df else pd.Series(False, index=df.index)
    out = add_fdr(df[has], family_cols=["fdr_family"]) if has.any() else df[has]
    rest = df[~has].copy()
    if len(rest):
        rest["p_fdr"] = np.nan
        rest["significant_fdr05"] = False
    return pd.concat([out, rest]).sort_index()


def _init(config):
    global OUT, PATHS, ART_DIR, ANS, CONDS, CORRECT_LETTER, KEYL, KEYS, KSET, LOG, LOG100, METRIC_KEY, MODELS, ORDER, QI, QS, QSI, REPORT_PATH, SUBS, bench, inv, runs
    out = config["propagation_tables"]
    OUT = Path(out["output_dir"])
    reports_root = Path(config["outputs"]["formal_reports"])
    variants_root = Path(config["outputs"]["report_models_source_ablation"])
    det = config["numerical_matching"]["deterministic_conditions"]
    PATHS = dict(benchmark=Path(config["benchmark"]["source_csv"]),
                 runs=Path(config["result_tables"]["output_dir"]) / "runs.csv",
                 inventory=Path(config["numerical_matching"]["outputs"]["match_inventory_csv"]),
                 audit_logs={lv: Path(det[f"num{lv}"]["mutation_log"]) for lv in ["010", "025", "050", "100"]},
                 reports=lambda c: (variants_root / "reports" if c.startswith(("dsv4_", "q35_")) else reports_root) / f"{c}.json",
                 ev_styles=Path(config["outputs"]["corruption_direct_evidence"]) / "texts" / "ev_styles.json",
                 ev_numeric_changes=Path(config["outputs"]["corruption_direct_evidence"]) / "logs" / "ev_numeric_changes.json",
                 coverage_logs=Path(config["paths"]["output_root"]) / "coverage_rebuild",
                 top1_removal=variants_root / "logs" / "top1_removal.json",
                 direct_evidence={c: Path(config["direct_rag"]["outputs"]["evidence_dir"]) / f"{c}.jsonl" for c in ["rag_clean", "rag_num100"]})
    ART_DIR = Path(config["corpora"]["clean"])
    bench = pd.read_csv(used(PATHS["benchmark"]), encoding="utf-8-sig")
    ORDER = ["RadioRAG", "RaR", "New-31"]
    KEYS = sorted(((r.dataset, int(r.q_number)) for r in bench.itertuples()), key=lambda k: (ORDER.index(k[0]), k[1]))
    QI = {k: i for i, k in enumerate(KEYS)}
    QS = [f"{d}|{q}" for d, q in KEYS]
    QSI = {s: i for i, s in enumerate(QS)}
    KEYL = {(r.dataset, int(r.q_number)): "ABCDE"[int(r.answer_index) - 1] for r in bench.itertuples()}
    CORRECT_LETTER = np.array([KEYL[k] for k in KEYS])
    runs = pd.read_csv(used(PATHS["runs"]), keep_default_na=False)
    MODELS = sorted(runs.model.unique())
    CONDS = sorted(runs.condition.unique())
    ANS = {}
    for (m, c), g in runs.groupby(["model", "condition"]):
        validate_question_keys(g.to_dict("records"), KEYS, f"{m} {c}")
        A = dict(letter=np.array([""] * 200, dtype=object), parsed=np.zeros(200), correct=np.zeros(200), conf=np.full(200, np.nan),
                 hr=np.zeros(200), unsafe=np.zeros(200), contra=np.zeros(200), ev=np.array([""] * 200, dtype=object))
        for r in g.itertuples():
            i = QI[(r.dataset, int(r.q_number))]
            A["letter"][i] = r.answer_letter
            A["parsed"][i] = r.answer_letter != ""
            A["correct"][i] = int(r.correct)
            A["conf"][i] = float(r.confidence) if r.confidence != "" else np.nan
            wrong = int(r.correct) == 0
            A["hr"][i] = int(wrong and str(r.chosen_high_risk) == "1")
            A["unsafe"][i] = int(wrong and str(r.chosen_unsafe) == "1")
            A["contra"][i] = int(wrong and str(r.chosen_contradicts) == "1")
            A["ev"][i] = r.evidence_sha256
        ANS[(m, c)] = A
    assert len(ANS) == 7 * 42
    METRIC_KEY = {"accuracy": "correct", "high_risk_error_rate": "hr", "unsafe_error_rate": "unsafe", "contradiction_error_rate": "contra"}


    inv = pd.read_csv(used(PATHS["inventory"]), encoding="utf-8-sig", usecols=["condition", "dataset", "q_number", "article_id", "start_char", "end_char", "original", "perturbed",
                                                                     "value_changed", "substring_match", "complete_number_match", "original_complete_number_match", "match_spans_json"])
    inv["q"] = inv.dataset + "|" + inv.q_number.astype(str)
    inv["mk"] = inv.article_id + "|" + inv.start_char.astype(str) + "|" + inv.end_char.astype(str)
    inv["chg"] = inv.value_changed == 1
    inv["mod"] = (inv.complete_number_match == 1) & inv.chg
    inv["orig"] = (inv.original_complete_number_match == 1) & inv.chg
    inv["sub"] = (inv.substring_match == 1) & inv.chg
    LOG, KSET = {}, {}
    for lv in ["010", "025", "050", "100"]:
        d = pd.DataFrame(json.load(open(used(PATHS["audit_logs"][lv]))))
        d["mk"] = d.article_id + "|" + d.start_char.astype(str) + "|" + d.end_char.astype(str)
        LOG[lv] = d
        KSET[lv] = set(d.mk)
    assert KSET["010"] <= KSET["025"] <= KSET["050"] <= KSET["100"]
    LOG100 = LOG["100"].drop_duplicates("mk").set_index("mk")
    REPORT_PATH = {c: PATHS["reports"](c) for c in CONDS if c not in ("closedbook", "rag_clean", "rag_num100") and not (c.startswith("ev_"))}
    assert len(REPORT_PATH) == 32


    SUBS = {"all": np.ones(200, bool)}


def vals(m, c, met):
    return ANS[(m, c)][METRIC_KEY[met]]


def changed(m, a, b):
    A, B = ANS[(m, a)], ANS[(m, b)]
    return ((A["parsed"] == 1) & (B["parsed"] == 1) & (A["letter"] != B["letter"])).astype(float)


def pooled(fn):
    return np.mean([fn(m) for m in MODELS], axis=0)


def reference(c):
    if c.startswith("topk"):
        return None if c == "topk05_pad00" else "topk05_pad00"
    if c in ("ev_conflict", "ev_numeric", "ev_forum", "ev_overconfident", "ev_hedged", "ev_review"):
        return "ev_clean"
    if c == "rag_clean":
        return "closedbook"
    if c == "rag_num100":
        return "rag_clean"
    if c.startswith("dsv4_") and c != "dsv4_clean":
        return "dsv4_clean"
    if c.startswith("q35_") and c != "q35_clean":
        return "q35_clean"
    if c in ("clean", "ev_clean", "closedbook"):
        return None
    return "clean"


GROUPS = {
    "vs_clean": [(c, "clean") for c in ["num010", "num025", "num050", "num100", "cov010", "cov025", "cov050", "cov100", "noretrieval", "clean_r2", "clean_r3"]],
    "volume": [(c, "topk05_pad00") for c in ["topk01_pad00", "topk03_pad00", "topk10_pad00", "topk20_pad00", "topk05_pad50", "topk05_pad75", "topk05_pad90", "topk05_pad95"]],
    "vs_closedbook": [(c, "closedbook") for c in ["clean", "noretrieval", "ev_clean"]],
    "vs_ev_clean": [(c, "ev_clean") for c in ["ev_conflict", "ev_numeric", "ev_forum", "ev_overconfident", "ev_hedged", "ev_review"]],
    "top1": [("top1_100", "clean")],
    "report_models": [(p + s, p + "clean") for p in ["dsv4_", "q35_"] for s in ["clean_r2", "num100", "cov100", "noretrieval"]] + [("dsv4_clean", "clean"), ("q35_clean", "clean")],
    "direct_retrieval": [("rag_num100", "rag_clean"), ("rag_clean", "closedbook"), ("rag_clean", "clean"), ("rag_num100", "num100")],
}

def ids(items):
    return {(x["article_id"] if isinstance(x, dict) else x) for x in items}


def unit_class(s):
    s = str(s).lower()
    for name, pat in UNIT_CLASSES:
        if re.search(pat, s):
            return name
    return "other"


def reports(c):
    if c not in _REP:
        recs = json.load(open(used(REPORT_PATH[c]), encoding="utf-8"))
        assert len(recs) == 200, c
        validate_question_keys(recs, KEYS, c)
        _REP[c] = {f"{r['dataset']}|{int(r['q_number'])}": r for r in recs}
    return _REP[c]


def layout(c, q):
    if (c, q) not in _LAY:
        text = reports(c)[q]["report"]
        out, idx, i, n = [], [], 0, len(text)
        while i < n:
            if text[i].isspace():
                j = i
                while j < n and text[j].isspace():
                    j += 1
                out.append(" ")
                idx.append(i)
                i = j
            else:
                out.append(text[i])
                idx.append(i)
                i += 1
        normed = "".join(out)
        assert normed == WS.sub(" ", text)
        lines, pos, option, block, n_sec, closed = [], 0, "outside", "other", 0, True
        n_opt = len(reports(c)[q]["options"])
        for line in text.split("\n"):
            s = line.strip()
            if re.match(r"^##\s+Final note", s, re.I):
                option, block, closed = "outside", "other", True
            elif re.match(r"^##\s", s) and closed:
                option = "ABCDE"[n_sec] if n_sec < n_opt else "outside"
                n_sec, closed, block = n_sec + 1, False, "other"
            elif re.match(r"^###\s+Sources", s):
                block, closed = "sources", True
            else:
                low = s.lower().strip("#*: ")
                if low.startswith("supporting evidence"):
                    block = "supporting"
                elif low.startswith("contradictory evidence") or low.startswith("contradicting evidence"):
                    block = "contradictory"
                elif low.startswith("neutral synthesis"):
                    block = "neutral"
            lines.append((pos, pos + len(line), option, block, line))
            pos += len(line) + 1
        _LAY[(c, q)] = (normed, idx, lines)
    return _LAY[(c, q)]


def locate(c, q, spans_json, version):
    normed, idx, lines = layout(c, q)
    res = []
    for sp in json.loads(spans_json):
        if sp["version"] != version or sp["start"] >= len(idx):
            continue
        raw = idx[sp["start"]]
        for a, b, option, block, line in lines:
            if a <= raw <= b:
                res.append((option, block, line))
                break
    return res


def pairs(ref, comp, lv):
    key = (ref, comp, lv)
    if key not in _PAIRS:
        cols = ["q", "mk", "article_id", "original", "perturbed", "mod", "orig", "match_spans_json"]
        a = inv[(inv.condition == ref) & inv.chg & inv.mk.isin(KSET[lv])][cols]
        b = inv[(inv.condition == comp) & inv.chg & inv.mk.isin(KSET[lv])][cols]
        j = a.merge(b[["q", "mk", "mod", "orig", "match_spans_json"]], on=["q", "mk"], suffixes=("_a", "_b"))
        j["qi"] = j.q.map(QSI)
        j["den"] = j.orig_a & ~j.mod_a
        j["strict"] = j.den & j.mod_b & ~j.orig_b
        j["loose"] = j.mod_b & ~j.mod_a
        j["reverse"] = j.mod_a & ~j.orig_a & j.orig_b & ~j.mod_b
        _PAIRS[key] = j
    return _PAIRS[key]


def aligned(j, ref, comp):
    keep = []
    for r in j[j.strict].itertuples():
        oa = {o for o, _, _ in locate(ref, r.q, r.match_spans_json_a, "original")}
        ob = {o for o, _, _ in locate(comp, r.q, r.match_spans_json_b, "perturbed")}
        keep.append(bool(oa & ob))
    s = pd.Series(False, index=j.index)
    if keep:
        s.loc[j.index[j.strict]] = np.array(keep, dtype=bool)
    return s


STOP = set("with that this from have been were which their there than into also such most more about between these those other they when where while will would could should being both each some only over under after before usually often commonly typically generally approximately".split())


def source_sentence(article_id, mk, original):
    if article_id not in _ART:
        _ART[article_id] = open(used(ART_DIR / article_id), encoding="utf-8", newline="").read()
    start, end = int(mk.split("|")[-2]), int(mk.split("|")[-1])
    raw = _ART[article_id]
    for text in (raw, raw.replace("\r\n", "\n")):
        if text[start:end] == original:
            a0 = max(text.rfind("\n", 0, start), text.rfind(". ", 0, start)) + 1
            b0 = min([x for x in (text.find("\n", end), text.find(". ", end)) if x != -1] or [len(text)])
            return " ".join(text[a0:b0 + 1].split())
    return ""


def content_words(text):
    return set(re.findall(r"[a-z]{4,}", text.lower())) - STOP


def events(ref, comp, lv):
    key = (ref, comp, lv)
    if key not in _AUG:
        j = pairs(ref, comp, lv).copy()
        j["section_aligned"] = aligned(j, ref, comp)
        cov = pd.Series(0.0, index=j.index)
        for r in j[j.strict].itertuples():
            st = content_words(source_sentence(r.article_id, r.mk, r.original))
            lb = locate(comp, r.q, r.match_spans_json_b, "perturbed")
            cov[r.Index] = max([len(st & content_words(l)) / len(st) for _, _, l in lb] or [0.0]) if st else 0.0
        j["source_cov"] = cov
        j["source_matched_30"] = j.strict & (j.source_cov >= 0.3)
        j["source_matched_50"] = j.strict & (j.source_cov >= 0.5)
        _AUG[key] = j
    return _AUG[key]


DEFS = ["strict", "loose", "section_aligned", "source_matched_30", "source_matched_50"]


def qind(j, col):
    v = np.zeros(200)
    v[j.loc[j[col], "qi"].unique()] = 1
    return v


def comparisons(g):
    G = GEN[g]
    out = []
    for ref in G["clean"]:
        for lv, comp in G["changed"].items():
            out.append((ref, comp, lv, "changed"))
        for comp in G["clean"]:
            if comp != ref:
                for lv in G["changed"]:
                    out.append((ref, comp, lv, "clean_repeat"))
    return out


def build_propagation():
    prow, qrow, crow, loc, cau, vt = [], [], [], [], [], []
    narrow = re.compile(r"inconsisten|discrepan|conflict|contradict|erroneous|incorrect|implausib|questionable|doubt|unreliable|typo|misprint|mistake|caution|unverified|cannot be verified|differs from|at odds|atypical", re.I)
    for g, G in GEN.items():
        for ref, comp, lv, kind in comparisons(g):
            j = events(ref, comp, lv)
            den = j[j.den]
            base = dict(report_model=G["model"], reference=ref, comparison=comp, comparison_type=kind, level=int(lv))
            outcomes = {"replaced": den.mod_b & ~den.orig_b, "kept": den.orig_b & ~den.mod_b, "both": den.mod_b & den.orig_b, "neither": ~den.mod_b & ~den.orig_b}
            for name, ind in outcomes.items():
                prow.append(dict(**base, outcome=name, n_shared_pairs=len(j), n_original_stated=len(den), n_outcome=int(ind.sum()), n_questions=int(den.q.nunique()),
                                 **metric(ind.values, den.qi.values)))
            for definition in DEFS:
                ev = j[j[definition]]
                qrow.append(dict(**base, definition=definition, n_events=len(ev), n_distinct_values=int(ev.drop_duplicates(["q", "perturbed"]).shape[0]),
                                 n_reverse_events=int(j.reverse.sum()), n_questions_with_event=int(ev.q.nunique()), **metric(qind(j, definition))))
            if kind == "changed":
                for definition in DEFS:
                    for rep in [x for x in G["clean"] if x != ref]:
                        jr = events(ref, rep, lv)
                        crow.append(dict(**base, definition=definition, repeat=rep, contrast=f"{comp}_minus_{rep}_reference_{ref}", **pdiff(qind(j, definition), qind(jr, definition)),
                                         fdr_family=f"propagation|{definition}"))
            if ref != G["clean"][0]:
                continue
            ev = j[j.strict]
            rows = []
            for r in ev.itertuples():
                lb = locate(comp, r.q, r.match_spans_json_b, "perturbed")
                la = locate(ref, r.q, r.match_spans_json_a, "original")
                cl = CORRECT_LETTER[r.qi]
                rows.append(dict(qi=r.qi, q=r.q, mk=r.mk, in_correct=any(o == cl for o, _, _ in lb), in_other=any(o not in (cl, "outside") for o, _, _ in lb), outside=any(o == "outside" for o, _, _ in lb),
                                 **{f"block_{b}": any(x == b for _, x, _ in lb) for b in ["supporting", "contradictory", "neutral", "sources", "other"]},
                                 chg_narrow=any(narrow.search(l) for _, _, l in lb),
                                 org_narrow=any(narrow.search(l) for _, _, l in la), n_b=len(lb), n_a=len(la)))
            E = pd.DataFrame(rows)
            EVENTS[(g, lv, kind, comp)] = E
            if len(E):
                for kindc, cats in [("option", {"correct option": "in_correct", "other option": "in_other", "outside the option sections": "outside"}),
                                    ("block", {b: f"block_{b}" for b in ["supporting", "contradictory", "neutral", "sources", "other"]})]:
                    for cat, col in cats.items():
                        qv = np.zeros(200)
                        qv[E.loc[E[col], "qi"].unique()] = 1
                        loc.append(dict(**base, category_kind=kindc, category=cat, row_unit="event", n_events=len(E), n_in_category=int(E[col].sum()), **metric(E[col].values, E.qi.values)))
                        loc.append(dict(**base, category_kind=kindc, category=cat, row_unit="question", n_events=len(E), n_in_category=int(qv.sum()), **metric(qv)))
                for tl in ["narrow"]:
                    cau.append(dict(**base, statement="line with the changed value, comparison report", term_list=tl, n_events=len(E), n_with_term=int(E[f"chg_{tl}"].sum()), **metric(E[f"chg_{tl}"].values, E.qi.values)))
                    cau.append(dict(**base, statement="line with the original value, reference report", term_list=tl, n_events=len(E), n_with_term=int(E[f"org_{tl}"].sum()), **metric(E[f"org_{tl}"].values, E.qi.values)))
                    d = report_paired_diff(E[f"chg_{tl}"].values.astype(float), E[f"org_{tl}"].values.astype(float), "x", cluster_ids=E.qi.values)
                    cau.append(dict(**base, statement="changed value line minus original value line", term_list=tl, n_events=len(E), n_with_term=int(E[f"chg_{tl}"].sum()) - int(E[f"org_{tl}"].sum()), **unpack(d, "percentage_points")))
            if kind == "changed" and lv == "100":
                d2 = den.assign(replaced=(den.mod_b & ~den.orig_b).values, unit_class=den.original.map(unit_class).values)
                for cat, sub in [("all", d2)] + sorted(d2.groupby("unit_class"), key=lambda t: -len(t[1])):
                    vt.append(dict(**base, class_kind="unit_class", value_class=cat, quantity="share replaced", n_original_stated=len(sub), n_replaced=int(sub.replaced.sum()), **metric(sub.replaced.values, sub.qi.values)))
                    qv = np.zeros(200)
                    qv[sub.loc[sub.replaced, "qi"].unique()] = 1
                    vt.append(dict(**base, class_kind="unit_class", value_class=cat, quantity="questions with a replacement of this class", n_original_stated=len(sub), n_replaced=int(qv.sum()), **metric(qv)))
    write(pd.DataFrame(prow), "propagation_pairs.csv")
    write(pd.DataFrame(qrow), "propagation_questions.csv")
    write(fdr(crow), "propagation_contrasts.csv")
    write(pd.DataFrame(loc), "propagation_location.csv")
    write(pd.DataFrame(cau), "propagation_wording.csv")
    write(pd.DataFrame(vt), "propagation_value_types.csv")


EVENTS = {}


def build_propagation_answers():
    rows = []
    for g, G in GEN.items():
        ref = G["clean"][0]
        for lv, comp in G["changed"].items():
            j = pairs(ref, comp, lv)
            E = EVENTS[(g, lv, "changed", comp)]
            R = qind(j, "strict") == 1
            RC = np.zeros(200, bool)
            if len(E):
                RC[E.loc[E.in_correct, "qi"].unique()] = True
            subsets = {"replaced": R, "not_replaced": ~R, "replaced_in_correct_option_section": RC}
            reps = [x for x in G["clean"] if x != ref]
            for sname, mask in subsets.items():
                if mask.sum() < 5:
                    continue
                for m in MODELS + ["all"]:
                    f = (lambda fn: fn(m)) if m != "all" else pooled
                    base = dict(report_model=G["model"], level=int(lv), subset=sname, n_questions=int(mask.sum()), model=m, condition_a=comp, condition_b=ref)
                    fam = f"propagation_answers|{g}|{sname}|{'pooled' if m == 'all' else 'models'}"
                    for met in ["accuracy", "high_risk_error_rate"]:
                        a, b = f(lambda mm: vals(mm, comp, met)), f(lambda mm: vals(mm, ref, met))
                        rows.append(dict(**base, metric=met, contrast=f"{comp}_minus_{ref}", **pdiff(a[mask], b[mask]), fdr_family=f"{fam}|{met}",
                                         level_a=float(100 * report_metric(a[mask], "x")["x_mean_raw"]), level_b=float(100 * report_metric(b[mask], "x")["x_mean_raw"])))
                    for rep in reps:
                        a, b = f(lambda mm: changed(mm, comp, ref)), f(lambda mm: changed(mm, rep, ref))
                        rows.append(dict(**base, metric="changed_answer_rate", contrast=f"change_{comp}_vs_{ref}_minus_change_{rep}_vs_{ref}", **pdiff(a[mask], b[mask]),
                                         fdr_family=f"{fam}|changed_answer_rate", level_a=float(100 * report_metric(a[mask], "x")["x_mean_raw"]), level_b=float(100 * report_metric(b[mask], "x")["x_mean_raw"])))
            if g == "main" and lv == "100":
                cnt = np.zeros(200)
                for qi, n in j[j.strict].groupby("qi").size().items():
                    cnt[qi] = n
                for label, mask in [("0", cnt == 0), ("1", cnt == 1), ("2", cnt == 2), ("3 or more", cnt >= 3)]:
                    base = dict(report_model=G["model"], level=int(lv), subset=f"replacements_per_question={label}", n_questions=int(mask.sum()), model="all", condition_a=comp, condition_b=ref)
                    a, b = pooled(lambda mm: vals(mm, comp, "accuracy")), pooled(lambda mm: vals(mm, ref, "accuracy"))
                    rows.append(dict(**base, metric="accuracy", contrast=f"{comp}_minus_{ref}", **pdiff(a[mask], b[mask]), fdr_family="propagation_answers|main|dose|accuracy",
                                     level_a=float(100 * report_metric(a[mask], "x")["x_mean_raw"]), level_b=float(100 * report_metric(b[mask], "x")["x_mean_raw"])))
                    a, b = pooled(lambda mm: changed(mm, comp, ref)), pooled(lambda mm: changed(mm, reps[0], ref))
                    rows.append(dict(**base, metric="changed_answer_rate", contrast=f"change_{comp}_vs_{ref}_minus_change_{reps[0]}_vs_{ref}", **pdiff(a[mask], b[mask]),
                                     fdr_family="propagation_answers|main|dose|changed_answer_rate", level_a=float(100 * report_metric(a[mask], "x")["x_mean_raw"]), level_b=float(100 * report_metric(b[mask], "x")["x_mean_raw"])))
    write(fdr(rows), "propagation_answers.csv")


def build_pooled():
    mrow, crow = [], []
    for c in CONDS:
        ref = reference(c)
        for s, mask in SUBS.items():
            v = {met: pooled(lambda mm: vals(mm, c, met)) for met in ["accuracy", "high_risk_error_rate", "unsafe_error_rate", "contradiction_error_rate"]}
            if ref is not None:
                v["changed_answer_rate"] = pooled(lambda mm: changed(mm, c, ref))
            for met, x in v.items():
                mrow.append(dict(model="all", condition=c, subset=s, metric=met, **metric(x[mask])))
            conf = np.array([ANS[(m, c)]["conf"] for m in MODELS]) / 100.0
            ok = mask & ~np.all(np.isnan(conf), axis=0)
            mrow.append(dict(model="all", condition=c, subset=s, metric="mean_confidence", **metric(np.nanmean(conf[:, ok], axis=0))))
    write(pd.DataFrame(mrow), "metrics_pooled.csv")
    for grp, prs in GROUPS.items():
        for a, b in prs:
            for s, mask in SUBS.items():
                for met in ["accuracy", "high_risk_error_rate", "unsafe_error_rate", "contradiction_error_rate"]:
                    crow.append(dict(model="all", subset=s, metric=met, contrast=f"{a}_minus_{b}", condition_a=a, condition_b=b,
                                     **pdiff(pooled(lambda mm: vals(mm, a, met))[mask], pooled(lambda mm: vals(mm, b, met))[mask]), fdr_family=f"{grp}|{met}|{s}"))
    for s, mask in SUBS.items():
        d1 = pooled(lambda mm: vals(mm, "rag_num100", "accuracy")) - pooled(lambda mm: vals(mm, "rag_clean", "accuracy"))
        d2 = pooled(lambda mm: vals(mm, "num100", "accuracy")) - pooled(lambda mm: vals(mm, "clean", "accuracy"))
        crow.append(dict(model="all", subset=s, metric="accuracy", contrast="rag_change_minus_report_change", condition_a="rag_num100", condition_b="num100", **pdiff(d1[mask], d2[mask]),
                         fdr_family=f"direct_retrieval|accuracy|{s}"))
    for rep in ["clean_r2", "clean_r3"]:
        for c in CONDS:
            ref = reference(c)
            if ref is None or c == rep or c in ("rag_clean",) or c.startswith("ev_") or c.startswith("topk"):
                continue
            if c.startswith(("dsv4_", "q35_")):
                pre = c.split("_")[0] + "_"
                if rep == "clean_r3" or c == pre + "clean_r2":
                    continue
                ba, bb = pre + "clean_r2", pre + "clean"
            elif c == "rag_num100":
                ba, bb = rep, "clean"
            else:
                ba, bb = rep, "clean"
            for s, mask in SUBS.items():
                crow.append(dict(model="all", subset=s, metric="changed_answer_rate", contrast=f"change_{c}_vs_{ref}_minus_change_{ba}_vs_{bb}", condition_a=c, condition_b=ref,
                                 **pdiff(pooled(lambda mm: changed(mm, c, ref))[mask], pooled(lambda mm: changed(mm, ba, bb))[mask]), fdr_family=f"changed_minus_{rep}|changed_answer_rate|{s}"))
    write(fdr(crow), "contrasts_pooled.csv")


def subset_rows(mrow, crow, sname, mask, cond_pairs, extra=None):
    for m in MODELS + ["all"]:
        f = (lambda fn: fn(m)) if m != "all" else pooled
        for met in ["accuracy", "high_risk_error_rate"]:
            for c in sorted({x for p in cond_pairs for x in p}):
                mrow.append(dict(model=m, condition=c, subset=sname, metric=met, n_questions=int(mask.sum()), **(extra or {}), **metric(f(lambda mm: vals(mm, c, met))[mask])))
            for a, b in cond_pairs:
                transitions = {}
                if met == "accuracy":
                    for label, before, after in [("correct_to_incorrect", 1, 0), ("incorrect_to_correct", 0, 1)]:
                        values = f(lambda mm: ((vals(mm, b, met) == before) & (vals(mm, a, met) == after)).astype(float))[mask]
                        transitions[f"{label}_mean_raw"] = report_metric(values, "x")["x_mean_raw"]
                crow.append(dict(model=m, subset=sname, metric=met, contrast=f"{a}_minus_{b}", condition_a=a, condition_b=b, n_questions=int(mask.sum()), **(extra or {}),
                                 **pdiff(f(lambda mm: vals(mm, a, met))[mask], f(lambda mm: vals(mm, b, met))[mask]),
                                 **transitions,
                                 fdr_family=f"subsets|{sname.split('@')[0]}|{met}|{'pooled' if m == 'all' else 'models'}"))


def build_subsets():
    mrow, crow = [], []
    ch = ANS[(MODELS[0], "ev_clean")]["ev"] != ANS[(MODELS[0], "ev_numeric")]["ev"]
    pr = [("ev_numeric", "ev_clean"), ("num100", "clean"), ("rag_num100", "rag_clean"), ("ev_clean", "closedbook"), ("clean", "closedbook")]
    subset_rows(mrow, crow, "ev_numeric_changed", ch, pr)
    subset_rows(mrow, crow, "ev_numeric_unchanged", ~ch, pr)
    st = json.load(open(used(PATHS["ev_styles"])))
    ok = {s: np.zeros(200, bool) for s in ["forum", "overconfident", "hedged", "review"]}
    for r in st:
        ok[r["style"]][QI[(r["dataset"], int(r["q_number"]))]] = r["content_check_2"] == "YES"
    for s, mask in ok.items():
        if mask.sum() >= 5:
            subset_rows(mrow, crow, f"style_accepted@{s}", mask, [(f"ev_{s}", "ev_clean")])
    allfour = ok["forum"] & ok["overconfident"] & ok["hedged"] & ok["review"]
    names = ["ev_forum", "ev_overconfident", "ev_hedged", "ev_review"]
    subset_rows(mrow, crow, "style_accepted", allfour, [(a, b) for i, a in enumerate(names) for b in names[i + 1:]])
    STYLE_OK.update({s: int(m.sum()) for s, m in ok.items()})
    STYLE_OK["all_four"] = int(allfour.sum())
    write(pd.DataFrame(mrow), "metrics_subsets.csv")
    write(fdr(crow), "contrasts_subsets.csv")


STYLE_OK = {}


def build_confidence():
    rows = []
    lst = [(g, a, b) for g, prs in GROUPS.items() for a, b in prs]
    for grp, a, b in lst:
        diffs = []
        for m in MODELS:
            ca, cb = ANS[(m, a)]["conf"] / 100.0, ANS[(m, b)]["conf"] / 100.0
            ok = ~np.isnan(ca) & ~np.isnan(cb)
            rows.append(dict(model=m, metric="mean_confidence", contrast=f"{a}_minus_{b}", condition_a=a, condition_b=b, **pdiff(ca[ok], cb[ok]), fdr_family=f"{grp}|mean_confidence|models"))
            d = np.where(ok, ca - cb, np.nan)
            diffs.append(d)
        d = np.array(diffs)
        ok = ~np.all(np.isnan(d), axis=0)
        dm = np.nanmean(d[:, ok], axis=0)
        rows.append(dict(model="all", metric="mean_confidence", contrast=f"{a}_minus_{b}", condition_a=a, condition_b=b, **pdiff(dm, np.zeros_like(dm)), fdr_family=f"{grp}|mean_confidence|pooled"))
    st = json.load(open(used(PATHS["ev_styles"])))
    ok = {x: np.zeros(200, bool) for x in ["forum", "overconfident", "hedged", "review"]}
    for r in st:
        ok[r["style"]][QI[(r["dataset"], int(r["q_number"]))]] = r["content_check_2"] == "YES"
    for x, mask in ok.items():
        diffs = []
        for m in MODELS:
            ca, cb = ANS[(m, f"ev_{x}")]["conf"] / 100.0, ANS[(m, "ev_clean")]["conf"] / 100.0
            diffs.append(np.where(~np.isnan(ca) & ~np.isnan(cb) & mask, ca - cb, np.nan))
        d = np.array(diffs)
        keep = ~np.all(np.isnan(d), axis=0)
        dm = np.nanmean(d[:, keep], axis=0)
        rows.append(dict(model="all", metric="mean_confidence", contrast=f"ev_{x}_minus_ev_clean_on_accepted_rewrites", condition_a=f"ev_{x}", condition_b="ev_clean", **pdiff(dm, np.zeros_like(dm)),
                         fdr_family="style_accepted|mean_confidence|pooled"))
    write(fdr(rows), "contrasts_confidence.csv")


def build_reports():
    top = json.load(open(used(PATHS["top1_removal"])))
    removed_top1 = ids(top["removed_articles"])
    topart = {}
    for r in top["per_question"]:
        v = r.get("top1_article") or r.get("selected_article") or r.get("removed_article")
        topart[f"{r['dataset']}|{int(r['q_number'])}"] = v if isinstance(v, str) else (v.get("article_id") if isinstance(v, dict) else str(v))
    dele = {lv: ids(json.load(open(used(PATHS["coverage_logs"] / f"coverage_{lv}_deletions.json")))["deleted_articles"]) for lv in ["010", "025", "050", "100"]}
    changed_articles = {lv: set(LOG[lv].article_id) for lv in LOG}
    ev = {}
    for r in runs[runs.model == MODELS[0]].itertuples():
        ev[(r.condition, f"{r.dataset}|{int(r.q_number)}")] = r.evidence_sha256
    icount = inv[inv.chg].groupby(["condition", "q"]).agg(n_candidate_values=("chg", "size"), n_substring_found=("sub", "sum"), n_complete_found=("mod", "sum"), n_original_complete_found=("orig", "sum"))
    rows = []
    for c in sorted(REPORT_PATH):
        pre = "dsv4_" if c.startswith("dsv4_") else "q35_" if c.startswith("q35_") else ""
        rm = DSV4 if pre == "dsv4_" else Q35 if pre == "q35_" else MAIN
        base = c[len(pre):]
        refc = pre + "clean"
        R, RR = reports(c), reports(refc)
        lv = base[3:] if base.startswith(("num", "cov")) else None
        for q in QS:
            r = R[q]
            sha = hashlib.sha256(r["report"].encode("utf-8")).hexdigest()
            ret = {x["article_id"] for e in r["retrieval_log"] for x in e.get("results", [])}
            retref = {x["article_id"] for e in RR[q]["retrieval_log"] for x in e.get("results", [])}
            row = dict(condition=c, report_model=rm, dataset=q.split("|")[0], q_number=int(q.split("|")[1]), report_status=r["report_status"], attempts=r["attempts"],
                       n_queries=len(r["retrieval_log"]), n_results=sum(len(e.get("results", [])) for e in r["retrieval_log"]), n_retrieved=len(ret),
                       n_retrieved_in_reference=len(ret & retref) if c != refc else np.nan, jaccard_with_reference=(len(ret & retref) / len(ret | retref) if (ret | retref) else np.nan) if c != refc else np.nan,
                       report_words=len(r["report"].split()), report_chars=len(r["report"]), evidence_hash_matches_answers=int(ev.get((c, q)) == sha),
                       max_tool_output_chars=max([max(x.get("tool_output_chars") or [0]) for x in r.get("report_model_calls", [])] or [0]))
            lvn = lv if base.startswith("num") else ("100" if base in ("clean", "clean_r2", "clean_r3") else None)
            if lvn:
                row["n_perturbed_retrieved"] = len(ret & changed_articles[lvn])
            if (c, q) in icount.index:
                row.update(icount.loc[(c, q)].to_dict())
            if pre == "":
                row["top_cited_clean_article"] = topart.get(q, "")
            if base.startswith("cov"):
                row["n_deleted_retrieved"] = len(ret & dele[lv])
                row["n_reference_retrieved_deleted"] = len(retref & dele[lv])
                if pre == "":
                    row["top_cited_removed"] = int(topart.get(q) in dele[lv])
            if base == "top1_100":
                row["n_deleted_retrieved"] = len(ret & removed_top1)
                row["top_cited_removed"] = int(topart.get(q) in removed_top1)
                row["n_top1_retrieved"] = int(topart.get(q) in ret)
            rows.append(row)
    df = pd.DataFrame(rows)
    for g, G in GEN.items():
        for comp in list(G["changed"].values()) + G["clean"][1:]:
            lv = [k for k, v in G["changed"].items() if v == comp]
            j = pairs(G["clean"][0], comp, lv[0] if lv else "100")
            cnt = j[j.strict].groupby("q").size()
            sel = df.condition == comp
            df.loc[sel, "n_replacements"] = df.loc[sel].apply(lambda r: cnt.get(f"{r.dataset}|{r.q_number}", 0), axis=1)
    write(df, "reports.csv")
    cv = []
    pool_n = sum(1 for path in ART_DIR.glob("*.txt") if path.is_file())
    for lv in ["010", "025", "050", "100"]:
        d = LOG[lv]
        cv.append(dict(variant=f"numerical_{lv}", pool_n=pool_n, changed_n=int(d.article_id.nunique()), values_changed_n=int((d.original != d.perturbed).sum()), values_unchanged_after_rounding_n=int((d.original == d.perturbed).sum())))
    pool = json.load(open(used(PATHS["coverage_logs"] / "coverage_relevant_articles.json")))
    for lv in ["010", "025", "050", "100"]:
        cv.append(dict(variant=f"coverage_{lv}", pool_n=int(pool["relevant_pool_size"]), changed_n=len(dele[lv])))
    en = pd.DataFrame(json.load(open(used(PATHS["ev_numeric_changes"]))))
    cv.append(dict(variant="ev_numeric", pool_n=200, changed_n=int(en.article_id.nunique()), values_changed_n=int((en.original != en.perturbed).sum()), values_unchanged_after_rounding_n=int((en.original == en.perturbed).sum())))
    for s in ["forum", "overconfident", "hedged", "review"]:
        cv.append(dict(variant=f"ev_{s}", pool_n=200, changed_n=200, accepted_n=STYLE_OK.get(s), failed_n=200 - STYLE_OK.get(s, 0)))
    cv.append(dict(variant="top1_100", pool_n=pool_n, changed_n=int(top["unique_removed_articles"]), failed_n=int(top["unresolved_source_count"])))
    write(pd.DataFrame(cv), "corpus_variants.csv")
    num = ["n_queries", "n_retrieved", "report_words", "jaccard_with_reference", "n_reference_retrieved_deleted", "n_deleted_retrieved", "n_replacements", "attempts"]
    srow = []
    for c, gdf in df.groupby("condition"):
        for col in num:
            x = gdf[col].dropna().astype(float) if col in gdf else pd.Series(dtype=float)
            if len(x):
                srow.append(dict(condition=c, quantity=col, n=len(x), mean=x.mean(), sd=x.std(ddof=1), median=x.median(), q1=x.quantile(0.25), q3=x.quantile(0.75), minimum=x.min(), maximum=x.max()))
    write(pd.DataFrame(srow), "reports_summary.csv")


def build_direct_exposure():
    def load(c):
        out = {}
        with open(used(PATHS["direct_evidence"][c]), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                r["passages"] = []
                for p in r["selected_passages"]:
                    header = p["block_text"][:-len(p["selected_passage"])]
                    block = p["included_block_text"]
                    r["passages"].append({"article_id": p["article_id"], "included_text": block[len(header):] if len(block) >= len(header) else ""})
                out[f"{r['dataset']}|{int(r['q_number'])}"] = r
        return out
    C, N = load("rag_clean"), load("rag_num100")
    changed_articles = set(LOG["100"].loc[LOG["100"].original != LOG["100"].perturbed, "article_id"])
    rows = []
    for q in QS:
        pc = {(p["article_id"] if p["article_id"].endswith(".txt") else p["article_id"] + ".txt"): p["included_text"] for p in C[q]["passages"]}
        pn = {(p["article_id"] if p["article_id"].endswith(".txt") else p["article_id"] + ".txt"): p["included_text"] for p in N[q]["passages"]}
        n_sup, n_art = 0, 0
        for a, tn in pn.items():
            if a not in changed_articles:
                continue
            n_art += 1
            n_sup += supplied_changes(a, tn)
        rows.append(dict(dataset=q.split("|")[0], q_number=int(q.split("|")[1]), n_passages=len(pn), n_same_articles=len(set(pc) & set(pn)), n_changed_articles_supplied=n_art, n_changed_values_supplied=n_sup))
    E = pd.DataFrame(rows)
    write(E, "direct_retrieval_exposure.csv")
    exp = (E.n_changed_values_supplied > 0).values
    mrow = []
    crow = []
    for sname, mask in [("supplied_changed_value", exp), ("no_supplied_changed_value", ~exp)]:
        if mask.sum() >= 5:
            subset_rows(mrow, crow, f"{sname}@rag_num100", mask, [("rag_num100", "rag_clean")])
    write(pd.DataFrame(mrow), "direct_retrieval_exposure_metrics.csv")
    write(fdr(crow), "direct_retrieval_exposure_contrasts.csv")


def source_record(article):
    if article not in _PREFIX_RECORDS:
        original = Path(used(ART_DIR / article)).read_text(encoding="utf-8")
        rows = LOG["100"][LOG["100"].article_id == article]
        entries = [MutationEntry(article, r.original, r.perturbed, int(r.start_char), int(r.end_char), i, {})
                   for i, r in enumerate(rows.itertuples())]
        reconstruction = reconstruct_article(original, entries)
        lo = len(original) - len(original.lstrip())
        lm = len(reconstruction.modified_text) - len(reconstruction.modified_text.lstrip())
        inside, partial = {}, set()
        for entry in entries:
            ex = source_expressions_source_entry(original, reconstruction, entry)
            key = (int(entry.source_start), int(entry.source_end))
            if ex["partial_source"]:
                partial.add(key)
            inside[key] = (ex["status"] == "resolved" and ex["original_expression_start"] >= lo
                           and ex["modified_expression_start"] >= lm and ex["original_expression_end"] - lo <= 2000
                           and ex["modified_expression_end"] - lm <= 2000)
        _PREFIX_RECORDS[article] = (reconstruction, entries, inside, partial)
    return _PREFIX_RECORDS[article]


def supplied_changes(article, delivered):
    reconstruction, entries, _, _ = source_record(article)
    if not delivered:
        return 0
    for crlf in (False, True):
        convert = (lambda text: text.replace("\n", "\r\n")) if crlf else (lambda text: text)
        raw = convert(reconstruction.modified_text)
        if not raw.strip().startswith(delivered):
            continue
        leading = len(raw) - len(raw.lstrip())
        count = 0
        for entry in entries:
            if not value_changed_after_rounding(entry.original, entry.perturbed):
                continue
            start, end = reconstruction.entry_modified_spans[entry.entry_index]
            start = len(convert(reconstruction.modified_text[:start])) - leading
            changed = convert(entry.perturbed)
            end = start + len(changed)
            if 0 <= start < end <= len(delivered):
                if delivered[start:end] != changed:
                    raise ValueError(f"Changed span does not match the delivered text: {article}")
                count += 1
        return count
    raise ValueError(f"Cannot reconstruct the delivered passage: {article}")


def build_prefix_sensitivity():
    rows = []
    for group, generator in GEN.items():
        for ref, comp, level, kind in comparisons(group):
            matched = pairs(ref, comp, level).copy()
            matched["prefix"] = False
            matched["nonpartial"] = False
            for r in matched[matched.strict].itertuples():
                start, end = map(int, r.mk.rsplit("|", 2)[1:])
                matched.loc[r.Index, "prefix"] = source_record(r.article_id)[2][(start, end)]
                matched.loc[r.Index, "nonpartial"] = (start, end) not in source_record(r.article_id)[3]
            selected = matched[matched.prefix]
            rows.append(dict(report_model=generator["model"], reference=ref, comparison=comp, level=int(level), comparison_type=kind,
                             n_events=int(matched.strict.sum()), n_prefix_events=len(selected),
                             n_nonpartial_events=int(matched.nonpartial.sum()),
                             n_questions_without_partial=int(matched.loc[matched.nonpartial, "q"].nunique()),
                             n_questions_with_event=int(matched.loc[matched.strict, "q"].nunique()),
                             n_questions_with_prefix_event=int(selected.q.nunique()), **metric(qind(matched, "prefix"))))
    write(pd.DataFrame(rows), "propagation_prefix_sensitivity.csv")


def build_reporting_checks():
    missing = []
    for (model, condition), records in runs.groupby(["model", "condition"]):
        missing.append(dict(model=model, condition=condition, n=len(records),
                            unparsed=int((records.answer_letter == "").sum()),
                            missing_confidence=int((records.confidence == "").sum()),
                            length_finish=int((records.finish_reason == "length").sum())))
    write(pd.DataFrame(missing), "answer_completeness.csv")
    padding = []
    for condition in ["topk05_pad50", "topk05_pad75", "topk05_pad90", "topk05_pad95"]:
        relevant = padded = positive = 0
        shares = []
        for record in reports(condition).values():
            delivered = [result for call in record["retrieval_log"] for result in call["results"]
                         if result.get("included_in_context", True)]
            selected = [r for r in delivered if r.get("retrieval_role") == "irrelevant_padding"]
            relevant += len(delivered) - len(selected)
            padded += len(selected)
            positive += sum(float(r["score"]) > 0 for r in selected)
            shares.append(len(selected) / len(delivered) if delivered else 0)
        padding.append(dict(condition=condition, relevant_passages=relevant, padding_passages=padded,
                            positive_score_padding=positive, mean_question_share=float(np.mean(shares)),
                            min_question_share=min(shares), max_question_share=max(shares)))
    write(pd.DataFrame(padding), "retrieval_padding.csv")
    rows = []
    for group in ["vs_clean", "top1"]:
        for a, b in GROUPS[group]:
            av = pooled(lambda model: ((ANS[(model, a)]["hr"] == 1) | (ANS[(model, a)]["parsed"] == 0)).astype(float))
            bv = pooled(lambda model: ((ANS[(model, b)]["hr"] == 1) | (ANS[(model, b)]["parsed"] == 0)).astype(float))
            rows.append(dict(comparison=a, reference=b, metric="high_risk_or_unparsed",
                             **pdiff(av, bv), fdr_family=f"response_failure|{group}"))
    write(fdr(rows), "response_failure_sensitivity.csv")


def select_examples(events):
    selected = events[events.report_model == MAIN].copy()
    selected["o"] = selected.dataset.map({"RadioRAG": 0, "RaR": 1, "New-31": 2})
    for field, label in [("clean_report_line", "cc"), ("changed_report_line", "cx")]:
        selected[label] = [len(content_words(source) & content_words(line)) / len(content_words(source))
                           if content_words(source) else 0.0
                           for source, line in zip(selected.source_sentence, selected[field])]
    selected["mn"] = selected[["cc", "cx"]].min(axis=1)
    selected["sm"] = selected.cc + selected.cx
    selected = selected[(selected.same_option_section == 1) & (selected.clean_report_line.str.len() <= 300)
                        & (selected.changed_report_line.str.len() <= 300) & (selected.source_sentence.str.len() <= 220)]
    rows = []
    for unit in ["length", "percent", "time or age", "attenuation", "pressure", "volume"]:
        rows.append(selected[selected.unit_class == unit].sort_values(
            ["mn", "sm", "o", "q_number", "article_id", "original"],
            ascending=[False, False, True, True, True, True]).iloc[0])
    return pd.DataFrame(rows)


def build_examples():
    rows = []
    for g, G in GEN.items():
        ref, comp = G["clean"][0], G["changed"]["100"]
        j = events(ref, comp, "100")
        for r in j[j.strict].itertuples():
            lb = locate(comp, r.q, r.match_spans_json_b, "perturbed")
            la = locate(ref, r.q, r.match_spans_json_a, "original")
            cl = CORRECT_LETTER[r.qi]
            rows.append(dict(report_model=G["model"], dataset=r.q.split("|")[0], q_number=int(r.q.split("|")[1]), article_id=r.article_id, pattern_type=LOG100.loc[r.mk, "pattern_type"],
                             unit_class=unit_class(r.original), original=r.original, changed=r.perturbed, source_sentence=source_sentence(r.article_id, r.mk, r.original),
                             clean_report_line=la[0][2].strip() if la else "", changed_report_line=lb[0][2].strip() if lb else "", source_word_coverage=round(float(r.source_cov), 3),
                             same_option_section=int(r.section_aligned), option_sections=",".join(sorted({o for o, _, _ in lb})), correct_option=cl,
                             in_correct_option_section=int(any(o == cl for o, _, _ in lb))))
    events_frame = pd.DataFrame(rows)
    write(events_frame, "propagation_events.csv")
    write(select_examples(events_frame), "propagation_selected_examples.csv")


def run_propagation_tables(config):
    set_scale("percent")
    INPUTS.clear()
    for cache in (_REP, _LAY, _PAIRS, _ART, _AUG, _PREFIX_RECORDS, EVENTS, STYLE_OK):
        cache.clear()
    _init(config)
    OUT.mkdir(parents=True, exist_ok=True)
    build_examples()
    build_propagation()
    build_propagation_answers()
    build_pooled()
    build_subsets()
    build_confidence()
    build_reports()
    build_direct_exposure()
    build_prefix_sensitivity()
    build_reporting_checks()
    for p in list(INPUTS):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        INPUTS[p] = h.hexdigest()
    params = dict(module="src/propagation_tables.py", module_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
                  n_boot=1000, boot_seed=0, scale="percent", central_estimator="bootstrap_mean",
                  binary_test="exact_mcnemar", inputs=INPUTS,
                  inference_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (Path(__file__).resolve().parents[1] / "Inference").glob("*.py")})
    tmp = OUT / f"build_params_propagation_tables.{os.getpid()}.tmp"
    json.dump(params, open(tmp, "w"), indent=1)
    os.replace(tmp, OUT / "build_params_propagation_tables.json")
    return OUT


def main_propagation_tables(config_path):
    return run_propagation_tables(load_config(config_path))
