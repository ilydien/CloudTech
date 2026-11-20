"""Konvensi nama tabel & partisi medallion, plus helper kalender bursa.

Semua path relatif terhadap satu namespace key (lihat ``ihsg.storage``):

    bronze/yf/prices/ticker=JKSE/ingest_date=2026-11-13/data.parquet
    bronze/yf/actions/ticker=JKSE/ingest_date=2026-11-13/data.parquet
    bronze/yf/_manifest/ingest_date=2026-11-13/data.parquet
    silver/prices/ticker=JKSE/data.parquet
    silver/panel/as_of=2026-11-13/data.parquet
    silver/_dq/as_of=2026-11-13/data.parquet
    silver/_quarantine/as_of=2026-11-13/data.parquet
    gold/features/as_of=2026-11-13/data.parquet
    gold/predictions/as_of=2026-11-13/data.parquet
    gold/regime/as_of=2026-11-13/data.parquet
    gold/regime_eval/as_of=2026-11-13/data.parquet
    gold/vol_eval/as_of=2026-11-13/data.parquet
    gold/metrics/as_of=2026-11-13/data.parquet
    gold/run_log/as_of=2026-11-13/stage=train/run_id=<id>/data.parquet
    gold/params/as_of=2026-11-13/data.parquet
    gold/champion/data.parquet
    mlartifacts/run_id=<id>/model.pt

Aturan penting:
* ``bronze`` append-only: satu partisi per ``ingest_date``, tidak pernah ditimpa.
* ``silver`` dan ``gold`` adalah turunan, boleh ditimpa penuh (idempoten).
* Semua file Parquet. Tidak ada CSV/JSON sebagai media penyimpanan.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from ihsg.config import JAKARTA_TZ, Settings

# --- prefix tabel -----------------------------------------------------------
BRONZE_PRICES = "bronze/yf/prices"
BRONZE_ACTIONS = "bronze/yf/actions"
BRONZE_MANIFEST = "bronze/yf/_manifest"

SILVER_PRICES = "silver/prices"
SILVER_PANEL = "silver/panel"
SILVER_DQ = "silver/_dq"
SILVER_QUARANTINE = "silver/_quarantine"

GOLD_FEATURES = "gold/features"
GOLD_PREDICTIONS = "gold/predictions"
GOLD_CANDIDATES = "gold/candidates"
GOLD_PUBLISHED = "gold/published"
GOLD_REGIME = "gold/regime"
GOLD_REGIME_EVAL = "gold/regime_eval"
GOLD_VOL_EVAL = "gold/vol_eval"
GOLD_METRICS = "gold/metrics"
GOLD_RUN_LOG = "gold/run_log"
GOLD_PARAMS = "gold/params"
GOLD_DQ = "gold/_dq"

GOLD_CHAMPION = "gold/champion/data.parquet"
ARTIFACTS = "mlartifacts"

PART_FILE = "data.parquet"

_INGEST_DATE_RE = re.compile(r"ingest_date=(\d{4}-\d{2}-\d{2})/")
_AS_OF_RE = re.compile(r"as_of=(\d{4}-\d{2}-\d{2})/")


# --- kunci tabel ------------------------------------------------------------
def bronze_prices_key(ticker_key: str, ingest_date: str) -> str:
    return f"{BRONZE_PRICES}/ticker={ticker_key}/ingest_date={ingest_date}/{PART_FILE}"


def bronze_actions_key(ticker_key: str, ingest_date: str) -> str:
    return f"{BRONZE_ACTIONS}/ticker={ticker_key}/ingest_date={ingest_date}/{PART_FILE}"


def bronze_manifest_key(ingest_date: str) -> str:
    return f"{BRONZE_MANIFEST}/ingest_date={ingest_date}/{PART_FILE}"


def silver_prices_key(ticker_key: str) -> str:
    return f"{SILVER_PRICES}/ticker={ticker_key}/{PART_FILE}"


def as_of_key(table: str, as_of: str) -> str:
    """Kunci standar untuk tabel gold yang dipartisi per tanggal acuan."""
    return f"{table}/as_of={as_of}/{PART_FILE}"


def artifact_key(run_id: str, name: str) -> str:
    return f"{ARTIFACTS}/run_id={run_id}/{name}"


# --- penemuan partisi -------------------------------------------------------
def existing_ingest_dates(sink, prefix: str) -> set[str]:
    """Tanggal ``ingest_date`` yang sudah punya part file di ``prefix``."""
    keys = sink.list_keys(f"{prefix}/")
    return {m.group(1) for key in keys if (m := _INGEST_DATE_RE.search(key))}


def existing_as_of(sink, table: str) -> list[str]:
    """Daftar ``as_of`` yang sudah ada untuk sebuah tabel gold (terurut)."""
    keys = sink.list_keys(f"{table}/")
    values = {m.group(1) for key in keys if (m := _AS_OF_RE.search(key))}
    return sorted(values)


# --- kalender ---------------------------------------------------------------
def trading_days(start: str | date, end: str | date, holidays: tuple[str, ...] = ()) -> pd.DatetimeIndex:
    """Hari bursa (Senin-Jumat, dikurangi hari libur) dalam rentang inklusif."""
    return pd.bdate_range(start=start, end=end, freq="C", holidays=list(holidays))


def is_trading_day(day: str, holidays: tuple[str, ...] = ()) -> bool:
    return len(trading_days(day, day, holidays)) == 1


def next_trading_day(day: str, holidays: tuple[str, ...] = ()) -> str:
    start = date.fromisoformat(day) + timedelta(days=1)
    days = trading_days(start, start + timedelta(days=15), holidays)
    if len(days) == 0:
        raise RuntimeError(f"Tidak ada hari bursa setelah {day} dalam 15 hari")
    return days[0].date().isoformat()


def resolve_as_of(settings: Settings, now: datetime | None = None) -> str:
    """Hari bursa terakhir yang sesinya sudah selesai.

    Sengaja TIDAK memakai tanggal hari ini: data penutupan hari ini baru final
    setelah pasar tutup, sementara job berjalan pagi. Dengan aturan ini, run
    pada 05:15 WIB tanggal 14 memakai ``as_of = 2026-11-13``.
    """
    moment = now or datetime.now(ZoneInfo(JAKARTA_TZ))
    today = moment.date()
    window_start = today - timedelta(days=20)
    days = [
        day.date()
        for day in trading_days(window_start.isoformat(), today.isoformat(), settings.idx_holidays)
        if day.date() < today
    ]
    if not days:
        return (today - timedelta(days=1)).isoformat()
    return days[-1].isoformat()


def date_range(start: str, end: str):
    """Yield setiap tanggal ISO inklusif dari ``start`` sampai ``end``."""
    cursor = date.fromisoformat(start)
    stop = date.fromisoformat(end)
    while cursor <= stop:
        yield cursor.isoformat()
        cursor += timedelta(days=1)


def year_chunks(start: str, end: str):
    """Pecah rentang menjadi jendela per tahun kalender (backfill terbatas)."""
    cursor = date.fromisoformat(start)
    stop = date.fromisoformat(end)
    while cursor <= stop:
        chunk_end = min(date(cursor.year, 12, 31), stop)
        yield cursor.isoformat(), chunk_end.isoformat()
        cursor = chunk_end + timedelta(days=1)


def assert_trading_day(day: str, holidays: tuple[str, ...] = ()) -> None:
    if not is_trading_day(day, holidays):
        raise ValueError(f"{day} bukan hari bursa (cek IDX_HOLIDAYS)")
