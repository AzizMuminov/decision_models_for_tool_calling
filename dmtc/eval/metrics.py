"""Aggregate metrics: accuracy with bootstrap CIs, paired tests, calibration, cost."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy.stats import binomtest


def bootstrap_ci(values: Sequence[float], n: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> dict[str, float]:
    """Exact McNemar test on paired per-example correctness of two conditions."""
    b01 = sum(1 for x, y in zip(a, b) if not x and y)
    b10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = b01 + b10
    p = 1.0 if n == 0 else binomtest(b01, n, 0.5).pvalue
    return {"a_only": b10, "b_only": b01, "p_value": float(p)}


def ece(confidences: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    if len(c) == 0:
        return math.nan
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c > lo) & (c <= hi) if lo > 0 else (c >= lo) & (c <= hi)
        if m.any():
            total += m.mean() * abs(c[m].mean() - y[m].mean())
    return float(total)


def brier(p_yes: Sequence[float], y: Sequence[bool]) -> float:
    p = np.asarray(p_yes, dtype=float)
    return float(np.mean((p - np.asarray(y, dtype=float)) ** 2)) if len(p) else math.nan


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    f1s = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def cost_usd(row: dict[str, Any], prices: dict[str, Any]) -> float:
    """Provider-reported cost when present (OpenRouter), else token counts x configured prices."""
    reported = row.get("llm_cost_usd", 0.0) + row.get("dm_cost_usd", 0.0)
    if reported > 0:
        return reported
    llm = prices.get("llm", {})
    dm = prices.get("dm", {})
    return (
        row["llm_input_tokens"] / 1e6 * llm.get("input_per_mtok", 0.0)
        + row["llm_output_tokens"] / 1e6 * llm.get("output_per_mtok", 0.0)
        + row["dm_input_tokens"] / 1e6 * dm.get("input_per_mtok", 0.0)
        + row["dm_calls"] * dm.get("per_request", 0.0)
    )


def auroc(scores: Sequence[float], y: Sequence[bool]) -> float:
    """Threshold-free gate quality: P(score of a should-call item > score of a should-not item)."""
    pos = [s for s, t in zip(scores, y) if t]
    neg = [s for s, t in zip(scores, y) if not t]
    if not pos or not neg:
        return math.nan
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj
