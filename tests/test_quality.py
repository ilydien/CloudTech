from __future__ import annotations

import pandas as pd
import pytest

from ihsg import quality


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(
            ["2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13"]
        ),
        "close": [100.0, 101.0, 99.0, 102.0, 103.0],
        "volume": [10.0, 0.0, 0.0, 0.0, 0.0],
        "ret_cc_pct": [0.0, 1.0, -2.0, 3.0, 1.0],
    })


def test_not_null_check_passes_for_complete_data():
    result = quality.check_not_null(_frame(), "t", "silver", ["date", "close"])
    assert result.passed


def test_positive_check_fails_on_zero_price():
    frame = _frame()
    frame.loc[0, "close"] = 0.0
    result = quality.check_positive(frame, "t", "silver", "close")
    assert not result.passed
    with pytest.raises(quality.QualityError):
        result.raise_if_hard()


def test_calendar_gap_is_soft_not_hard():
    frame = _frame().drop(index=2)  # hilangkan Rabu
    result = quality.check_calendar_gaps(frame, "t", "silver", ())
    assert result.severity == quality.SOFT
    assert result.failed_count == 1
    result.raise_if_hard()  # tidak boleh raise


def test_freshness_fails_when_data_is_stale():
    result = quality.check_freshness(_frame(), "t", "silver", "2026-11-20", max_lag_days=1)
    assert not result.passed
    assert result.severity == quality.HARD


def test_zero_volume_share_flags_forex_like_series():
    result = quality.check_zero_volume_share(_frame(), "t", "silver", threshold=0.5)
    assert result.failed_count == 1
    assert result.severity == quality.SOFT


def test_zero_volume_share_tolerates_ordinary_equity():
    result = quality.check_zero_volume_share(_frame(), "t", "silver", threshold=0.9)
    assert result.failed_count == 0


def test_report_frame_matches_contract():
    results = [
        quality.check_not_null(_frame(), "silver.prices", "silver", ["close"]),
        quality.check_calendar_gaps(_frame(), "silver.prices", "silver", ()),
    ]
    frame = quality.report_frame(results, "2026-11-13")
    assert list(frame.columns) == [
        "as_of", "layer", "table", "check_name", "severity",
        "passed", "failed_count", "total_count", "details",
    ]
    # details berupa 'k=v;k=v', bukan JSON
    assert not frame["details"].str.contains(r"[{}]").any()


def test_quarantine_frame_is_empty_when_no_failures():
    frame = _frame()
    result = quality.quarantine_frame(
        frame, pd.Series([False] * len(frame)), "reason", "silver", "t", "2026-11-13",
        key_column="date", value_column="close",
    )
    assert result.empty


def test_quarantine_frame_records_reason_and_value():
    frame = _frame()
    mask = frame["close"] < 100.0
    result = quality.quarantine_frame(
        frame, mask, "invalid_ohlc", "silver", "t", "2026-11-13",
        key_column="date", value_column="close",
    )
    assert len(result) == 1
    assert result["reason"].iloc[0] == "invalid_ohlc"
    assert set(result.columns) == {
        "as_of", "layer", "table", "reason", "key_value", "column_name", "column_value",
    }
