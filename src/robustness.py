"""Stratified (confounder-robustness) comparisons for observational cohorts.

``total_ads`` (how many ads a user saw) is a suspected confounder: exposure
influences conversion AND is itself influenced by being in the ad group, so a
pooled z-test can credit the treatment with what exposure produces. Splitting
into strata shows whether an observed advantage is uniform across exposure
levels or concentrated in a few buckets.

Honesty contract for this module: within-stratum comparisons are exploratory
robustness checks, not a pre-registered test. Sparse strata (expected event
counts below 5) get NaN z/p/CI because the normal approximation is not
trustworthy there; they are reported descriptively only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.inference import two_proportion_ztest, wald_ci_difference

# Exposure buckets chosen so the control (psa) group has enough support in
# each cell; the wide high bins reflect the long tail of total_ads (95th
# percentile is 88 ads), not arbitrary preference.
STRATUM_BINS: list[tuple[int, int]] = [
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 5),
    (6, 10),
    (11, 20),
    (21, 50),
    (50, 10**9),
]
STRATUM_LABELS: list[str] = ["1", "2", "3", "4-5", "6-10", "11-20", "21-50", "50+"]


def strata_table(
    df: pd.DataFrame,
    stratum_col: str = "total_ads",
    group_col: str = "test_group",
    outcome_col: str = "converted",
    group_levels: tuple[str, str] = ("ad", "psa"),
    bins: list[tuple[int, int]] = STRATUM_BINS,
    labels: list[str] = STRATUM_LABELS,
) -> pd.DataFrame:
    """Per-stratum two-group comparison with difference, CI and z-test.

    Returns a ready-to-display DataFrame; group_levels is (treatment,
    control) so diff_pp = p_treatment - p_control. Strata where the expected
    event count drops below 5 in either group carry NaN z/p/CI (the normal
    approximation fails) and sparse=True — treat those rows as descriptive
    only.
    """
    rows: list[dict[str, object]] = []
    for label, (lo, hi) in zip(labels, bins):
        sub = df.loc[df[stratum_col].between(lo, hi)]
        counts = {
            level: (
                int((sub[group_col] == level).sum()),
                int(sub.loc[sub[group_col] == level, outcome_col].sum()),
            )
            for level in group_levels
        }
        treat, control = group_levels
        n1, x1 = counts[treat]
        n2, x2 = counts[control]
        if n1 == 0 or n2 == 0:
            continue
        p1, p2 = x1 / n1, x2 / n2
        diff = p1 - p2
        expected = min(n1 * p1, n1 * (1 - p1), n2 * p2, n2 * (1 - p2))
        sparse = bool(expected < 5)
        if sparse:
            z = p_value = ci_low = ci_high = np.nan
        else:
            z, p_value = two_proportion_ztest(x1, n1, x2, n2)
            ci_low, ci_high = wald_ci_difference(p1, p2, n1, n2)
        rows.append(
            {
                "stratum": label,
                f"n_{treat}": n1,
                f"n_{control}": n2,
                f"x_{treat}": x1,
                f"x_{control}": x2,
                f"conversion_{treat}": p1,
                f"conversion_{control}": p2,
                "diff_pp": diff * 100,
                "ci_low_pp": ci_low * 100 if not sparse else np.nan,
                "ci_high_pp": ci_high * 100 if not sparse else np.nan,
                "z": z,
                "p_value": p_value,
                "sparse": sparse,
            }
        )
    return pd.DataFrame(rows)


def pooled_diff_excluding(
    df: pd.DataFrame,
    stratum_col: str,
    drop_lo: int | None = None,
    drop_hi: int | None = None,
    group_col: str = "test_group",
    outcome_col: str = "converted",
    group_levels: tuple[str, str] = ("ad", "psa"),
) -> dict[str, float]:
    """Pooled difference after dropping a stratum range, e.g. 50+ ads.

    Answers the driver check: how much of the pooled advantage comes from one
    exposure bucket? Returns diff_pp, z, p_value, ci_low_pp, ci_high_pp on the
    remaining rows; None bounds mean no rows are dropped.
    """
    keep = df
    if drop_lo is not None and drop_hi is not None:
        keep = df.loc[~df[stratum_col].between(drop_lo, drop_hi)]
    treat, control = group_levels
    n1, x1 = (
        int((keep[group_col] == treat).sum()),
        int(keep.loc[keep[group_col] == treat, outcome_col].sum()),
    )
    n2, x2 = (
        int((keep[group_col] == control).sum()),
        int(keep.loc[keep[group_col] == control, outcome_col].sum()),
    )
    p1, p2 = x1 / n1, x2 / n2
    z, p_value = two_proportion_ztest(x1, n1, x2, n2)
    ci_low, ci_high = wald_ci_difference(p1, p2, n1, n2)
    return {
        "diff_pp": (p1 - p2) * 100,
        "z": z,
        "p_value": p_value,
        "ci_low_pp": ci_low * 100,
        "ci_high_pp": ci_high * 100,
    }