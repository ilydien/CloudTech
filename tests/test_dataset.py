from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from ihsg import dataset
from ihsg.config import Settings

JAKARTA = ZoneInfo("Asia/Jakarta")


def _settings(**overrides) -> Settings:
    base = Settings.from_env()
    return base if not overrides else base.__class__(**{**base.__dict__, **overrides})


def test_trading_days_excludes_weekend():
    days = dataset.trading_days("2026-11-09", "2026-11-15")
    assert [d.date().isoformat() for d in days] == [
        "2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13",
    ]


def test_trading_days_respects_holidays():
    days = dataset.trading_days("2026-11-09", "2026-11-13", ("2026-11-11",))
    assert "2026-11-11" not in {d.date().isoformat() for d in days}
    assert len(days) == 4


def test_resolve_as_of_uses_previous_session():
    settings = _settings()
    now = datetime(2026, 11, 14, 5, 15, tzinfo=JAKARTA)  # Sabtu pagi
    assert dataset.resolve_as_of(settings, now) == "2026-11-13"


def test_resolve_as_of_skips_holiday():
    settings = _settings(idx_holidays=("2026-11-13",))
    now = datetime(2026, 11, 14, 5, 15, tzinfo=JAKARTA)
    assert dataset.resolve_as_of(settings, now) == "2026-11-12"


def test_next_trading_day_across_weekend():
    assert dataset.next_trading_day("2026-11-13") == "2026-11-16"


def test_partition_keys_are_stable():
    assert (
        dataset.bronze_prices_key("JKSE", "2026-11-13")
        == "bronze/yf/prices/ticker=JKSE/ingest_date=2026-11-13/data.parquet"
    )
    assert (
        dataset.as_of_key(dataset.GOLD_PREDICTIONS, "2026-11-13")
        == "gold/predictions/as_of=2026-11-13/data.parquet"
    )


def test_existing_ingest_dates_parses_partitions(tmp_path):
    from ihsg.storage import LocalSink

    sink = LocalSink(tmp_path)
    sink.write_parquet(
        pd.DataFrame({"Date": [pd.Timestamp("2026-11-12")], "Close": [1.0]}),
        dataset.bronze_prices_key("JKSE", "2026-11-12"),
    )
    sink.write_parquet(
        pd.DataFrame({"Date": [pd.Timestamp("2026-11-13")], "Close": [2.0]}),
        dataset.bronze_prices_key("JKSE", "2026-11-13"),
    )
    found = dataset.existing_ingest_dates(sink, f"{dataset.BRONZE_PRICES}/ticker=JKSE")
    assert found == {"2026-11-12", "2026-11-13"}


def test_year_chunks_cover_range():
    chunks = list(dataset.year_chunks("2024-06-01", "2026-02-01"))
    assert chunks[0] == ("2024-06-01", "2024-12-31")
    assert chunks[-1] == ("2026-01-01", "2026-02-01")
