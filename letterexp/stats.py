"""The statistics the letter reports, in one place."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy import stats


def wilson(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float, float]:
    """(p, lo, hi) Wilson 95% interval for k successes in n."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> Tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    ci = stats.binomtest(int(k), int(n)).proportion_ci(confidence_level=conf, method="exact")
    return k / n, float(ci.low), float(ci.high)


def fisher(k1: int, n1: int, k2: int, n2: int) -> Tuple[float, float]:
    """(odds ratio, two-sided p) for failures k1/n1 vs k2/n2."""
    res = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
    return float(res.statistic), float(res.pvalue)


def welch(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    """(mean(a) - mean(b), t, two-sided p), unequal variances."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = stats.ttest_ind(a, b, equal_var=False)
    return float(a.mean() - b.mean()), float(t.statistic), float(t.pvalue)


def chi2_counts(ks: Sequence[int], ns: Sequence[int]) -> Tuple[float, int, float, bool]:
    """Chi-square test of equal proportions across groups.

    Returns (chi2, dof, p, exact_used). With an expected cell below 5 the
    asymptotic test is unreliable, so a Monte-Carlo permutation p-value
    (20k draws, fixed seed) is returned instead and flagged.
    """
    ks, ns = np.asarray(ks, int), np.asarray(ns, int)
    table = np.vstack([ks, ns - ks])
    if table.sum(axis=1).min() == 0:
        return 0.0, len(ks) - 1, 1.0, False
    chi2, p, dof, exp = stats.chi2_contingency(table, correction=False)
    if exp.min() >= 5:
        return float(chi2), int(dof), float(p), False
    rng = np.random.default_rng(12345)
    pooled = np.concatenate([np.ones(ks.sum()), np.zeros((ns - ks).sum())])
    edges = np.cumsum(ns)[:-1]
    hits = 0
    draws = 20_000
    for _ in range(draws):
        rng.shuffle(pooled)
        kk = np.array([g.sum() for g in np.split(pooled, edges)])
        t = np.vstack([kk, ns - kk])
        c, *_ = stats.chi2_contingency(t, correction=False)
        hits += c >= chi2 - 1e-12
    return float(chi2), int(dof), (hits + 1) / (draws + 1), True


def fmt_rate(k: int, n: int) -> str:
    p, lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * p:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"
