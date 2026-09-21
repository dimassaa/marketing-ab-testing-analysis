"""Data integrity and EDA helpers for the marketing A/B analysis.

Golden standard #1 (data first): every downstream statistical claim is only
as trustworthy as the dataframe it runs on, so loading and validation are
kept here as reusable, testable functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

COLUMN_MAP: dict[str, str] = {
    "user id": "user_id",
    "test group": "test_group",
    "converted": "converted",
    "total ads": "total_ads",
    "most ads day": "most_ads_day",
    "most ads hour": "most_ads_hour",
}

# Expected values, used by the sanity checks (from the Kaggle dataset schema).
VALID_GROUPS = {"ad", "psa"}


def load_marketing_ab(path: str | Path) -> pd.DataFrame:
    """Load the marketing A/B CSV into a cleaned, typed dataframe.

    Drops the row-index column the Kaggle dump adds, renames columns to
    snake_case and casts columns to sensible dtypes so downstream code never
    has to guess. Returns the raw cleaned frame without any validation.
    """
    df = pd.read_csv(path)
    # "Unnamed: 0" is a stray row index from the Kaggle export — not data.
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = df.rename(columns=COLUMN_MAP)
    # converted is a 0/1 outcome; keep it as int64 for arithmetic.
    # No NaN survives this cast, so we validate for NaN separately below.
    df["converted"] = df["converted"].astype(int)
    return df


def integrity_report(df: pd.DataFrame) -> dict[str, object]:
    """Return an immutable-style summary of integrity checks as a dict.

    Covers: row/user counts and duplicated user ids, missing values per
    column, group composition and the converted 0/1 distribution. The caller
    (notebook or terminal run) renders this; no decision is taken here.
    """
    n_duplicate_user_ids = int(df["user_id"].duplicated().sum())
    missing = df.isna().sum()
    missing = missing[missing > 0] if missing.any() else pd.Series(dtype="int64")
    return {
        "n_rows": int(len(df)),
        "n_unique_users": int(df["user_id"].nunique()),
        "n_duplicate_user_ids": n_duplicate_user_ids,
        "duplicate_user_rows": int(df["user_id"].duplicated(keep=False).sum()),
        "missing_per_column": missing,
        "test_group_counts": df["test_group"].value_counts().to_dict(),
        "converted_value_counts": df["converted"].value_counts().sort_index().to_dict(),
    }


def assert_integrity(df: pd.DataFrame) -> None:
    """Fail loudly if the data cannot support a valid experiment.

    Raises ValueError with the exact offending values. Duplicated user ids are
    reported by integrity_report but are only fatal if the user decides to
    treat them as separate units — so they are not rejected here.
    """
    if df["test_group"].isna().any():
        raise ValueError("test_group contains missing values")
    unknown_groups = set(df["test_group"].unique()) - VALID_GROUPS
    if unknown_groups:
        raise ValueError(f"test_group contains unexpected values: {unknown_groups}")
    not_binary = set(df["converted"].unique()) - {0, 1}
    if not_binary:
        raise ValueError(f"converted must be 0/1, found: {not_binary}")


def conversion_rates(df: pd.DataFrame, group_col: str = "test_group") -> pd.DataFrame:
    """Overall and per-group conversion table used across EDA and reports.

    Returns a tidy frame with group, users and converted share, precomputed
    so notebooks do not re-derive the same aggregation in ten places.
    """
    agg = (
        df.groupby(group_col)["converted"]
        .agg(users="count", converted="sum")
        .assign(**{"conversion_rate": lambda d: d["converted"] / d["users"]})
        .reset_index()
    )
    return agg


def describe_frame(df: pd.DataFrame) -> Mapping[str, object]:
    """Compact descriptive snapshot of a cleaned frame for the EDA notebook.

    Includes dtypes, describe() of numerics and the top of most_ads_day, the
    three quantities the EDA narrative references.
    """
    return {
        "dtypes": df.dtypes.astype(str).to_dict(),
        "total_ads_describe": df["total_ads"].describe().to_dict(),
        "most_ads_day_counts": df["most_ads_day"].value_counts().to_dict(),
    }