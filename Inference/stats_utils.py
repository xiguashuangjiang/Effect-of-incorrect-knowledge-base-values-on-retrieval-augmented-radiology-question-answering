"""
Inference/stats_utils.py
Created on July 14, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

from typing import List, Optional

import numpy as np

N_BOOT    = 1000
BOOT_SEED = 0


def _checked_clusters(cluster_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(cluster_ids)
    if ids.dtype.kind in "fc":
        bad = ~np.isfinite(ids)
    else:
        flat = ids.astype(str)
        bad = np.isin(np.char.strip(np.char.lower(flat)), ("", "nan", "none", "<na>"))
    n_bad = int(bad.sum())
    if n_bad:
        raise ValueError(
            f"[stats] {n_bad} of {len(ids)} rows carry no cluster id. The resampling unit is the "
            "cluster, and a missing id would collapse every affected row into one cluster, which "
            "silently changes both n and the interval. Fix the cluster ids instead of dropping the "
            "check.")
    return ids


def _cluster_sums_setup(values: np.ndarray, cluster_ids: np.ndarray):
    unique_c, inverse = np.unique(_checked_clusters(cluster_ids), return_inverse=True)
    K = len(unique_c)
    counts = np.bincount(inverse, minlength=K).astype(float)
    sums = np.bincount(inverse, weights=values, minlength=K)
    return K, inverse, counts, sums


def _cluster_bootstrap_means(
    value_arrays: List[np.ndarray], cluster_ids: np.ndarray, n_boot: int, seed: int,
) -> List[np.ndarray]:
    K, inverse, counts, _ = _cluster_sums_setup(value_arrays[0], cluster_ids)
    rng = np.random.RandomState(seed)
    draw = rng.randint(0, K, size=(n_boot, K))
    total_count = counts[draw].sum(axis=1)
    results = []
    for values in value_arrays:
        sums = np.bincount(inverse, weights=values, minlength=K)
        total_sum = sums[draw].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            results.append(total_sum / total_count)
    return results


def bootstrap_proportion(
    values: np.ndarray,
    cluster_ids: Optional[np.ndarray] = None,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": 0}
    if cluster_ids is not None:
        boot_means, = _cluster_bootstrap_means([values], np.asarray(cluster_ids), n_boot, seed)
    else:
        rng = np.random.RandomState(seed)
        idx = rng.randint(0, n, size=(n_boot, n))
        boot_means = values[idx].mean(axis=1)
    return {
        "point":    float(values.mean()),
        "bootstrap_mean": float(boot_means.mean()),
        "std":      float(boot_means.std(ddof=1)),
        "ci_lower": float(np.percentile(boot_means, 2.5)),
        "ci_upper": float(np.percentile(boot_means, 97.5)),
        "n":        int(n),
    }


def paired_bootstrap_diff(
    values_a: np.ndarray,
    values_b: np.ndarray,
    cluster_ids: Optional[np.ndarray] = None,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    n = len(values_a)
    assert len(values_b) == n, "Both arrays must have equal length."
    if n == 0:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "p_value": np.nan, "n": 0}
    point = float(values_a.mean() - values_b.mean())
    if cluster_ids is not None:
        mean_a, mean_b = _cluster_bootstrap_means(
            [values_a, values_b], np.asarray(cluster_ids), n_boot, seed
        )
        boot_diffs = mean_a - mean_b
    else:
        rng = np.random.RandomState(seed)
        idx = rng.randint(0, n, size=(n_boot, n))
        boot_diffs = values_a[idx].mean(axis=1) - values_b[idx].mean(axis=1)
    ci_lower   = float(np.percentile(boot_diffs, 2.5))
    ci_upper   = float(np.percentile(boot_diffs, 97.5))
    std        = float(boot_diffs.std(ddof=1))
    centered   = boot_diffs - point
    p_value    = float(np.mean(np.abs(centered) >= abs(point)))
    p_value    = max(p_value, 1.0 / n_boot)
    return {
        "point":    point,
        "bootstrap_mean": float(boot_diffs.mean()),
        "std":      std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value":  p_value,
        "n":        int(n),
    }


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    if n == 0:
        return np.array([])
    nan_mask  = np.isnan(p_values)
    valid_idx = np.where(~nan_mask)[0]
    if len(valid_idx) == 0:
        return p_values.copy()
    valid_p  = p_values[valid_idx]
    m        = len(valid_p)
    order    = np.argsort(valid_p)
    p_sorted = valid_p[order]
    adj = p_sorted * m / (np.arange(m, dtype=float) + 1)
    for i in range(m - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    adj = np.minimum(adj, 1.0)
    inv_order = np.empty(m, dtype=int)
    inv_order[order] = np.arange(m)
    result = p_values.copy()
    result[valid_idx] = adj[inv_order]
    return result
