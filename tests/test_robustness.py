"""Tests for src/robustness.py (stratified robustness analysis).

Covers the strata-table mechanics on a synthetic frame where the expected
values are known exactly, plus integration spot-checks on the real dataset
and the driver check that exposed the non-uniformity of the ad effect.
"""

import numpy as np
import pandas as pd
import pytest

from src.robustness import pooled_diff_excluding, strata_table


def _synthetic_balanced() -> pd.DataFrame:
    """Two strata, equal group sizes, known rates; no sparse cells."""
    rng = np.random.default_rng(7)
    rows = []
    for stratum, p_treat, p_ctrl in [(1, 0.05, 0.02), (2, 0.15, 0.10)]:
        for group, p in (("ad", p_treat), ("psa", p_ctrl)):
            n = 4000
            x = rng.binomial(n, p)
            rows.append(
                pd.DataFrame(
                    {
                        "test_group": [group] * n,
                        "converted": np.r_[np.ones(x), np.zeros(n - x)],
                        "total_ads": [stratum] * n,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def test_strata_table_computes_expected_differences() -> None:
    df = _synthetic_balanced()
    out = strata_table(df).set_index("stratum")
    expected_diff = {"1": 3.0, "2": 5.0}  # p.p.
    assert out.loc["1", "diff_pp"] == pytest.approx(expected_diff["1"], abs=0.6)
    assert out.loc["2", "diff_pp"] == pytest.approx(expected_diff["2"], abs=0.6)
    assert not out.loc["1", "sparse"] and not out.loc["2", "sparse"]
    # Both groups are present in every stratum, control first in the rows.
    assert out.loc["1", "n_ad"] == 4000 and out.loc["1", "n_psa"] == 4000


def test_strata_table_flags_sparse_cells_and_nans_test() -> None:
    # Tiny cell counts: normal approximation invalid -> NaN test columns.
    df = pd.DataFrame(
        {
            "test_group": ["ad"] * 50 + ["psa"] * 50,
            "converted": [1] * 2 + [0] * 48 + [1] * 0 + [0] * 50,
            "total_ads": [1] * 100,
        }
    )
    out = strata_table(df)
    assert bool(out.iloc[0]["sparse"]) is True
    assert np.isnan(out.iloc[0]["z"]) and np.isnan(out.iloc[0]["p_value"])


def test_strata_table_skips_empty_strata() -> None:
    df = pd.DataFrame(
        {
            "test_group": ["ad", "psa"] * 30,
            "converted": [1, 0] * 30,
            "total_ads": [2] * 60,
        }
    )
    out = strata_table(df)
    assert "1" not in out["stratum"].tolist()
    assert "2" in out["stratum"].tolist()


def test_pooled_diff_excluding_drops_stratum() -> None:
    df = _synthetic_balanced()
    # Stratum "1" is nearly flat (3 p.p.), stratum "2" is dominant (5 p.p.);
    # dropping the strong stratum must shrink the pooled difference.
    full = pooled_diff_excluding(df, "total_ads")
    without_strong = pooled_diff_excluding(df, "total_ads", drop_lo=2, drop_hi=2)
    assert full["diff_pp"] > without_strong["diff_pp"]


def test_real_data_spot_checks() -> None:
    """The reversal observed in the real dataset (verified by hand before
    the notebook was built): advantage is concentrated in 50+ ads."""
    from src.eda import load_marketing_ab

    df = load_marketing_ab("data/marketing_AB.csv")
    tab = strata_table(df)
    # Pooled diff ~ +0.77 p.p.; removing the 50+ stratum collapses it.
    pooled = pooled_diff_excluding(df, "total_ads")
    assert pooled["diff_pp"] == pytest.approx(0.77, abs=0.02)
    without_heavy = pooled_diff_excluding(df, "total_ads", drop_lo=50, drop_hi=10**9)
    assert without_heavy["diff_pp"] < 0.3
    # The 50+ bucket is the one that is significant AND not sparse.
    heavy = tab.loc[tab["stratum"] == "50+"]
    assert not heavy.iloc[0]["sparse"]
    assert heavy.iloc[0]["p_value"] < 0.05
    # Middle buckets show no significant ad-minus-psa difference (NaN means
    # sparse -> keep only the non-sparse middle range).
    mids = tab.loc[tab["stratum"].isin(["4-5", "6-10", "11-20"])]
    assert (mids["p_value"] > 0.05).all()
    assert tab.iloc[0]["stratum"] == "1" and tab.iloc[0]["sparse"]