"""Inferential statistics for the two-proportion A/B experiment.

All formulas are implemented here so notebooks never re-derive them inline:
the functions are unit-tested (tests/test_inference.py) and cross-checked
against statsmodels where a reference exists.

Notation: p0 = baseline (control) proportion, p1 = p0 + MDE = treatment
proportion; x are event counts, n group sizes.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

Z_95 = stats.norm.ppf(0.975)


def two_proportion_ztest(
    x1: int, n1: int, x2: int, n2: int
) -> tuple[float, float]:
    """Two-sided z-test for the difference of two proportions.

    Uses the pooled proportion (pooled estimate) under the null hypothesis,
    which is the standard (and most powerful) choice when testing equality.
    Returns (z, two-sided p-value).
    """
    p1, p2 = x1 / n1, x2 / n2
    p_pooled = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p_value)


def wald_ci_difference(
    p1: float, p2: float, n1: int, n2: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Two-sided Wald confidence interval for (p1 - p2).

    Wald uses separate (unpooled) variances because it estimates a confidence
    region, not an equality test. Returns (lower, upper).
    """
    z = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    return float(diff - z * se), float(diff + z * se)


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h, the effect size for proportions.

    Arc-sine transform stabilises the variance, so h behaves like a d for
    normal data: |h| ~ 0.2 small, 0.5 medium, 0.8 large effect.
    """
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def sample_size_proportions(
    p0: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Per-group sample size for a two-sided two-proportion z-test.

    Closed form for equal groups; p1 = p0 + mde. The formula balances the
    Type I error budget (alpha -> the z_{1-alpha/2} term, pooled variance)
    against the Type II budget (z_{1-beta}, separate variances). Returns the
    smallest integer n meeting the power target.
    """
    p1 = p0 + mde
    p_bar = (p0 + p1) / 2
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    numerator = (
        z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
        + z_beta * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
    )
    n = (numerator / mde) ** 2
    return int(np.ceil(n))


def power_for_proportions(
    p0: float, p1: float, n: int, alpha: float = 0.05
) -> float:
    """Achieved statistical power for given group size n.

    Inverts the sample-size equation for the z_{1-beta} term and maps it to
    a probability. For p1 <= p0 the power is ~alpha (a pointless experiment)
    rather than a negative number.
    """
    if p1 <= p0:
        return float(alpha)
    p_bar = (p0 + p1) / 2
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    numerator = abs(p1 - p0) * np.sqrt(n) - z_alpha * np.sqrt(
        2 * p_bar * (1 - p_bar)
    )
    denominator = np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
    return float(stats.norm.cdf(numerator / denominator))


def power_curve(
    p0: float,
    p1: float,
    n_grid: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Power across a grid of group sizes, for plotting (n -> power)."""
    return np.array([power_for_proportions(p0, p1, int(n), alpha) for n in n_grid])


def n_req_vs_mde(
    p0: float,
    mdes: np.ndarray,
    alpha: float = 0.05,
    power: float = 0.8,
) -> np.ndarray:
    """Required per-group n for an array of MDE values.

    Elementwise wrapper around sample_size_proportions so sensitivity charts
    (required n vs MDE) are a one-liner instead of an inline loop. The result
    is decreasing in MDE: smaller effects demand quadratically more data.
    """
    return np.array(
        [sample_size_proportions(p0, float(m), alpha, power) for m in mdes]
    )