"""
Inference/report_utils.py
Created on July 14, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from Inference.stats_utils import (
    N_BOOT,
    bootstrap_proportion, paired_bootstrap_diff, bh_fdr,
)


def fmt_pct(x: float) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("nan")
    v = float(x)
    if np.isfinite(v) and abs(v) > 2.0:
        raise ValueError(
            f"[report] {v!r} was passed to the percent formatter, but a value on its "
            f"natural scale cannot exceed 2 in magnitude (levels sit in [0, 1], "
            f"differences and cosine-scale readouts in [-2, 2]). This is the "
            f"signature of a value that was ALREADY multiplied by 100 upstream: "
            f"scaling happens in exactly one place, and this is it. Emit the raw "
            f"proportion and let the formatter scale it once.")
    return round(v * 100.0, 1)


SCALES = ("decimal", "percent")
_SCALE = {"name": None}


def set_scale(name: str) -> None:
    if name not in SCALES:
        raise ValueError(f"[report] scale {name!r} is not one of {SCALES}")
    _SCALE["name"] = name


def fmt_decimal(x: float) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("nan")
    v = float(x)
    if np.isfinite(v) and abs(v) > 2.0:
        raise ValueError(f"[report] {v!r} was passed to the decimal formatter, but a value on its natural scale cannot exceed 2 in magnitude; it was scaled upstream")
    return round(v, 3)


def fmt_metric(x: float) -> float:
    if _SCALE["name"] is None:
        raise ValueError("[report] call set_scale('decimal') or set_scale('percent') before the first row")
    return fmt_pct(x) if _SCALE["name"] == "percent" else fmt_decimal(x)


def fmt_one(x: float) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("nan")
    return round(float(x), 1)


def keep_p(p: float) -> float:
    if p is None:
        return float("nan")
    return float(p)


def _pack(prefix: str, point, std, lo, hi, n, is_percent: bool) -> Dict:
    f = fmt_metric if is_percent else fmt_one
    return {
        f"{prefix}_mean":     f(point),
        f"{prefix}_std":      f(std),
        f"{prefix}_ci_low":   f(lo),
        f"{prefix}_ci_high":  f(hi),
        f"{prefix}_mean_raw": float(point) if point is not None and np.isfinite(point) else float("nan"),
        f"{prefix}_std_raw":  float(std)   if std   is not None and np.isfinite(std)   else float("nan"),
        f"{prefix}_ci_low_raw":  float(lo) if lo    is not None and np.isfinite(lo)    else float("nan"),
        f"{prefix}_ci_high_raw": float(hi) if hi    is not None and np.isfinite(hi)    else float("nan"),
        "n": int(n) if n is not None and not (isinstance(n, float) and np.isnan(n)) else 0,
    }


def report_metric(
    values: np.ndarray,
    prefix: str = "value",
    is_percent: bool = True,
    cluster_ids: Optional[np.ndarray] = None,
    n_boot: int = N_BOOT,
) -> Dict:
    res = bootstrap_proportion(np.asarray(values, dtype=float),
                               cluster_ids=cluster_ids, n_boot=n_boot)
    out = _pack(prefix, res.get("bootstrap_mean", np.nan), res["std"], res["ci_lower"],
                res["ci_upper"], res["n"], is_percent)
    out[f"{prefix}_point_raw"] = res["point"]
    out[f"{prefix}_central_estimator"] = "bootstrap_mean"
    return out


def report_paired_diff(
    a: np.ndarray,
    b: np.ndarray,
    prefix: str = "diff",
    is_percent: bool = True,
    cluster_ids: Optional[np.ndarray] = None,
    n_boot: int = N_BOOT,
) -> Dict:
    res = paired_bootstrap_diff(np.asarray(a, float), np.asarray(b, float),
                                cluster_ids=cluster_ids, n_boot=n_boot)
    out = _pack(prefix, res.get("bootstrap_mean", np.nan), res["std"], res["ci_lower"],
                res["ci_upper"], res["n"], is_percent)
    out[f"{prefix}_point_raw"] = res["point"]
    out[f"{prefix}_central_estimator"] = "bootstrap_mean"
    a, b = np.asarray(a, float), np.asarray(b, float)
    binary = cluster_ids is None and len(a) > 0 and np.isin(a, [0, 1]).all() and np.isin(b, [0, 1]).all()
    if binary:
        discordant_a = int(((a == 1) & (b == 0)).sum())
        discordant_b = int(((a == 0) & (b == 1)).sum())
        total = discordant_a + discordant_b
        out["p_raw"] = float(binomtest(discordant_a, total, 0.5).pvalue) if total else 1.0
        out["test_method"] = "exact_mcnemar"
    else:
        out["p_raw"] = keep_p(res["p_value"])
        out["test_method"] = "paired_bootstrap"
    out["p_fdr"] = float("nan")
    return out


def add_fdr(
    df: pd.DataFrame,
    family_cols: Optional[List[str]] = None,
    p_col: str = "p_raw",
    out_col: str = "p_fdr",
) -> pd.DataFrame:
    df = df.copy()
    if p_col not in df.columns:
        return df
    if family_cols:
        present = [c for c in family_cols if c in df.columns]
        if not present:
            df[out_col] = bh_fdr(df[p_col].values)
        else:
            for _, idx in df.groupby(present, dropna=False).groups.items():
                sub = df.loc[idx, p_col].values
                df.loc[idx, out_col] = bh_fdr(sub)
    else:
        df[out_col] = bh_fdr(df[p_col].values)
    df["significant_fdr05"] = df[out_col].apply(
        lambda q: bool(q < 0.05) if pd.notna(q) else False
    )
    return df
