"""Unit tests for src/eda.py integrity and loading helpers."""

from pathlib import Path

import pandas as pd
import pytest

from src.eda import (
    VALID_GROUPS,
    assert_integrity,
    conversion_rates,
    integrity_report,
    load_marketing_ab,
)


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """A tiny frame mirroring the Kaggle schema, used for loader tests."""
    rows = [
        [0, 1, "ad", 0, 10, "Monday", 12],
        [1, 2, "ad", 1, 20, "Tuesday", 18],
        [2, 3, "psa", 0, 5, "Monday", 12],
        [3, 3, "psa", 1, 8, "Friday", 20],
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "Unnamed: 0",
            "user id",
            "test group",
            "converted",
            "total ads",
            "most ads day",
            "most ads hour",
        ],
    )
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return path


def test_load_drops_unnamed_and_renames(sample_csv: Path) -> None:
    df = load_marketing_ab(sample_csv)
    assert "Unnamed: 0" not in df.columns
    assert set(df.columns) >= {"user_id", "test_group", "converted"}
    assert df["converted"].dtype == "int64"


def test_load_casts_converted_to_int(sample_csv: Path) -> None:
    df = load_marketing_ab(sample_csv)
    assert set(df["converted"].unique()) == {0, 1}


def test_integrity_report_counts_duplicates(sample_csv: Path) -> None:
    df = load_marketing_ab(sample_csv)
    report = integrity_report(df)
    # user 3 appears twice in the fixture.
    assert report["n_unique_users"] == 3
    assert report["n_duplicate_user_ids"] == 1
    assert report["duplicate_user_rows"] == 2
    # psa users all missing? no: totals should match the fixture.
    assert report["test_group_counts"]["ad"] == 2
    assert report["test_group_counts"]["psa"] == 2


def test_assert_integrity_passes_for_clean_frame(sample_csv: Path) -> None:
    df = load_marketing_ab(sample_csv)
    assert_integrity(df)


def test_assert_integrity_rejects_bad_group(sample_csv: Path) -> None:
    df = load_marketing_ab(sample_csv)
    df.loc[0, "test_group"] = "banner"
    with pytest.raises(ValueError, match="unexpected values"):
        assert_integrity(df)


def test_assert_integrity_rejects_non_binary_converted(sample_csv: Path) -> None:
    df = load_marketing_ab(sample_csv)
    df.loc[0, "converted"] = 7
    with pytest.raises(ValueError, match="0/1"):
        assert_integrity(df)


def test_conversion_rates_is_well_formed(sample_csv: Path) -> None:
    df = load_marketing_ab(sample_csv)
    rates = conversion_rates(df)
    assert list(rates["test_group"]) == ["ad", "psa"]
    assert (rates["conversion_rate"].between(0, 1)).all()
    assert rates["conversion_rate"].sum() == pytest.approx(1.0)


def test_valid_groups_match_dataset_schema() -> None:
    assert VALID_GROUPS == {"ad", "psa"}