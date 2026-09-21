"""Unit tests for src/inference.py.

Cross-checks against statsmodels cover the closed-form formulas; behavioural
tests (monotonicity, direction, trivial cases) guard the implementation.
"""

import numpy as np
import pytest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from src.inference import (
    cohens_h,
    n_req_vs_mde,
    power_curve,
    power_for_proportions,
    sample_size_proportions,
    two_proportion_ztest,
    wald_ci_difference,
)

P0 = 0.0179  # observed psa baseline from Stage 1
MDE = 0.005


def test_zero_difference_gives_huge_p() -> None:
    z, p = two_proportion_ztest(x1=100, n1=5000, x2=100, n2=5000)
    assert abs(z) < 1e-9
    assert abs(p - 1.0) < 1e-9


def test_ztest_matches_statsmodels() -> None:
    x1, n1, x2, n2 = 247, 8000, 63, 6000
    z, p = two_proportion_ztest(x1, n1, x2, n2)
    from statsmodels.stats.proportion import proportions_ztest

    z_ref, p_ref = proportions_ztest(
        count=[x1, x2], nobs=[n1, n2], alternative="two-sided"
    )
    assert z == pytest.approx(z_ref, rel=1e-9)
    assert p == pytest.approx(p_ref, rel=1e-9)


def test_wald_ci_contains_observed_difference() -> None:
    p1, p2, n1, n2 = 0.032, 0.018, 20000, 8000
    lo, hi = wald_ci_difference(p1, p2, n1, n2)
    diff = p1 - p2
    assert lo <= diff <= hi


def test_wald_ci_widens_with_larger_alpha() -> None:
    p1, p2, n1, n2 = 0.032, 0.018, 20000, 8000
    lo95, hi95 = wald_ci_difference(p1, p2, n1, n2)
    lo90, hi90 = wald_ci_difference(p1, p2, n1, n2, alpha=0.10)
    # Width is hi - lo (positive); a 90% CI must be strictly narrower than 95%.
    assert (hi90 - lo90) < (hi95 - lo95)
    assert lo90 >= lo95 and hi90 <= hi95


def test_cohens_h_zero_for_equal_proportions() -> None:
    assert cohens_h(0.03, 0.03) == pytest.approx(0.0, abs=1e-12)
    assert cohens_h(0.06, 0.03) > 0
    assert cohens_h(0.03, 0.06) < 0


def test_sample_size_matches_statsmodels() -> None:
    n = sample_size_proportions(P0, MDE)
    es = proportion_effectsize(P0 + MDE, P0)
    n_ref = NormalIndPower().solve_power(
        effect_size=es,
        alpha=0.05,
        power=0.8,
        ratio=1,
        alternative="two-sided",
    )
    # Two standard but distinct approximations (closed-form binomial vs
    # Cohen's-h normal power model) differ by <0.5%; only relative agreement
    # is asserted on purpose.
    assert n == pytest.approx(np.ceil(n_ref), rel=0.01)


def test_sample_size_decreases_with_larger_mde() -> None:
    n_small_mde = sample_size_proportions(P0, 0.003)
    n_big_mde = sample_size_proportions(P0, 0.010)
    assert n_small_mde > n_big_mde


def test_sample_size_increases_with_strict_alpha_or_more_power() -> None:
    base = sample_size_proportions(P0, MDE)
    stricter_alpha = sample_size_proportions(P0, MDE, alpha=0.01)
    more_power = sample_size_proportions(P0, MDE, power=0.9)
    assert stricter_alpha > base
    assert more_power > base


def test_power_at_required_n_hits_target() -> None:
    n = sample_size_proportions(P0, MDE)
    assert power_for_proportions(P0, P0 + MDE, n) == pytest.approx(0.8, abs=0.01)


def test_power_curve_monotonic_and_bounded() -> None:
    grid = np.arange(1000, 30001, 1000)
    powers = power_curve(P0, P0 + MDE, grid)
    assert np.all(np.diff(powers) >= 0)
    assert powers.min() > 0.05 and powers.max() <= 1.0


def test_n_req_vs_mde_matches_scalar_formula() -> None:
    mdes = np.array([0.001, 0.005, 0.010])
    ns = n_req_vs_mde(P0, mdes)
    assert ns.tolist() == [sample_size_proportions(P0, m) for m in mdes]
    # Recognisable anchors from the sensitivity table in notebook 02.
    assert ns[0] == 283_523 and ns[1] == 12_547 and ns[2] == 3_512


def test_n_req_vs_mde_decreasing_with_mde() -> None:
    ns = n_req_vs_mde(P0, np.arange(0.001, 0.011, 0.001))
    assert np.all(np.diff(ns) < 0)