"""Silver: bersihkan, beri tipe, dan bangun panel lintas instrumen.

Silver TIDAK mengagregasi dan TIDAK memodelkan. Yang dilakukan:
1. gabungkan semua partisi bronze, buang duplikat (ambil ingest terbaru),
2. rename kolom ke snake_case, cast tipe, hitung return & varians Parkinson,
3. validasi kontrak (hard) + catat kualitas data (soft) + karantina baris rusak,
4. sejajarkan semua instrumen ke kalender bursa BEI dengan ``ffill`` terbatas.

Karena total data kecil (±2 MB), silver dibangun ulang penuh setiap run.
Konsekuensinya: silver selalu bisa diturunkan ulang dari bronze -> idempoten.
"""

from __future__ import annotations

import time

import pandas as pd

from ihsg import dataset, indicators
from ihsg.config import Settings
from ihsg.instruments import INDEX_KEY, Instrument, get_instruments
from ihsg.quality import (
    CheckResult,
    HARD,
    SOFT,
    check_calendar_gaps,
    check_freshness,
    check_not_null,
    check_outliers,
    check_positive,
    check_range,
    check_zero_volume_share,
    quarantine_frame,
    write_quarantine,
    write_report,
)
from ihsg.runlog import append_run
from ihsg.schemas import SILVER_PANEL, SILVER_PRICES, fingerprint, validate_contract
from ihsg.storage import Sink, get_sink

PRICE_FIELDS = ("open", "high", "low", "close", "adj_close", "volume", "ret", "park")

_RENAME = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


def _read_bronze(sink: Sink, ticker_key: str) -> tuple[pd.DataFrame, str | None]:
    """Gabungkan seluruh partisi bronze satu ticker, dedupe per tanggal."""
    prefix = f"{dataset.BRONZE_PRICES}/ticker={ticker_key}"
    ingest_dates = sorted(dataset.existing_ingest_dates(sink, prefix))
    if not ingest_dates:
        return pd.DataFrame(), None

    frames = []
    for ingest_date in ingest_dates:
        frame = sink.read_parquet(dataset.bronze_prices_key(ticker_key, ingest_date))
        frame["_ingest_date"] = ingest_date
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.sort_values(["Date", "_ingest_date"]).drop_duplicates("Date", keep="last")
    return raw.reset_index(drop=True), ingest_dates[-1]


def _build_prices(raw: pd.DataFrame, instrument: Instrument) -> pd.DataFrame:
    frame = raw.rename(columns=_RENAME).copy()
    frame["date"] = pd.to_datetime(frame["Date"]).dt.normalize()
    frame["ticker"] = instrument.ticker

    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")

    frame["ret_cc_pct"] = indicators.log_return_pct(frame["adj_close"])
    frame["park_var"] = indicators.parkinson_var(frame["high"], frame["low"])
    frame["source_ingest_date"] = frame["_ingest_date"].astype("string")

    columns = [
        "date", "ticker", "open", "high", "low", "close", "adj_close", "volume",
        "ret_cc_pct", "park_var", "source_ingest_date",
    ]
    frame = frame[columns].sort_values("date").drop_duplicates("date", keep="last")
    return frame.reset_index(drop=True)


def _checks(
    prices: pd.DataFrame, instrument: Instrument, settings: Settings, as_of: str
) -> list[CheckResult]:
    table = f"silver.prices[{instrument.key}]"

    # Calendar gap hanya bermakna bila kalender libur BEI diisi. Tanpa itu,
    # setiap libur nasional akan terhitung sebagai "gap" dan DQ jadi bising.
    if instrument.key == INDEX_KEY and settings.idx_holidays:
        gap_check = check_calendar_gaps(prices, table, "silver", settings.idx_holidays)
    else:
        gap_check = CheckResult(
            "silver",
            table,
            "calendar_gaps",
            "info",
            0,
            len(prices),
            {
                "skipped": (
                    "IDX_HOLIDAYS kosong" if instrument.key == INDEX_KEY else "bukan kalender BEI"
                )
            },
        )

    results = [
        check_not_null(prices, table, "silver", ["date", "close", "adj_close"]),
        check_positive(
            prices, table, "silver", "close",
            severity=HARD if instrument.key == INDEX_KEY else SOFT,
        ),
        gap_check,
        check_zero_volume_share(prices, table, "silver"),
        check_outliers(prices, table, "silver", "ret_cc_pct"),
        check_range(prices, table, "silver", "park_var", 0.0, 400.0),
    ]
    if instrument.key == INDEX_KEY:
        results.append(check_freshness(prices, table, "silver", as_of))
    return results


def _quarantine(prices: pd.DataFrame, as_of: str, table: str) -> pd.DataFrame:
    mask = (prices["high"] < prices["low"]) | (prices["adj_close"] <= 0) | prices["close"].isna()
    return quarantine_frame(
        prices, mask.fillna(False), "invalid_ohlc", "silver", table, as_of,
        key_column="date", value_column="close",
    )


def _build_panel(
    frames: dict[str, pd.DataFrame], settings: Settings, as_of: str
) -> pd.DataFrame:
    """Panel lebar pada kalender **sesi nyata IHSG**, ``ffill`` maksimal N hari.

    Kalender diturunkan dari tanggal yang benar-benar ada di data IHSG, bukan
    dari ``bdate_range``. Dengan begitu libur nasional tidak pernah menjadi baris
    kosong - dan tidak ada target yang hilang hanya karena ``IDX_HOLIDAYS``
    belum lengkap.
    """
    index_frame = frames[INDEX_KEY]
    idx = pd.DatetimeIndex(sorted(index_frame["date"].unique()))

    panel = pd.DataFrame({"date": idx})
    for key, frame in frames.items():
        series = frame.set_index("date")
        aligned = series.reindex(idx)
        was_missing = aligned["close"].isna()
        aligned = aligned.ffill(limit=settings.ffill_limit)

        panel[f"{key}_close"] = aligned["close"].to_numpy()
        panel[f"{key}_open"] = aligned["open"].to_numpy()
        panel[f"{key}_high"] = aligned["high"].to_numpy()
        panel[f"{key}_low"] = aligned["low"].to_numpy()
        panel[f"{key}_adj_close"] = aligned["adj_close"].to_numpy()
        panel[f"{key}_volume"] = aligned["volume"].to_numpy()
        panel[f"{key}_ret"] = aligned["ret_cc_pct"].to_numpy()
        panel[f"{key}_park"] = aligned["park_var"].to_numpy()
        panel[f"is_filled_{key}"] = (was_missing & aligned["close"].notna()).to_numpy()

    validate_contract(panel, SILVER_PANEL)
    return panel


def run(
    settings: Settings,
    as_of: str | None = None,
    *,
    trigger: str = "pipeline",
    sink: Sink | None = None,
) -> dict:
    settings.validate()
    sink = sink or get_sink(settings)
    as_of = as_of or settings.as_of or dataset.resolve_as_of(settings)
    instruments = get_instruments(settings)
    started = time.perf_counter()

    frames: dict[str, pd.DataFrame] = {}
    results: list[CheckResult] = []
    quarantines: list[pd.DataFrame] = []
    fingerprints: dict[str, str] = {}

    for instrument in instruments:
        raw, _ = _read_bronze(sink, instrument.key)
        if raw.empty:
            continue
        prices = _build_prices(raw, instrument)
        # Hormati as_of: silver untuk tanggal lampau harus memotong data setelahnya,
        # supaya backfill/retrain historis bisa direproduksi.
        prices = prices[prices["date"] <= pd.Timestamp(as_of)].reset_index(drop=True)
        if prices.empty:
            continue
        validate_contract(prices, SILVER_PRICES)

        for result in _checks(prices, instrument, settings, as_of):
            results.append(result)
            result.raise_if_hard()

        sink.write_parquet(prices, dataset.silver_prices_key(instrument.key), overwrite=True)
        frames[instrument.key] = prices
        fingerprints[instrument.key] = fingerprint(prices)
        quarantines.append(_quarantine(prices, as_of, f"silver.prices[{instrument.key}]"))

    if INDEX_KEY not in frames:
        raise RuntimeError(f"data IHSG ({INDEX_KEY}) tidak tersedia di bronze; jalankan ingest dulu")

    panel = _build_panel(frames, settings, as_of)
    sink.write_parquet(panel, dataset.as_of_key(dataset.SILVER_PANEL, as_of), overwrite=True)

    non_empty = [frame for frame in quarantines if frame is not None and not frame.empty]
    quarantine = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
    write_report(sink, results, as_of, layer="silver")
    write_quarantine(sink, quarantine, as_of, layer="silver")

    panel_fingerprint = fingerprint(panel)
    duration = time.perf_counter() - started
    append_run(
        sink,
        as_of=as_of,
        run_id=f"silver-{panel_fingerprint[:12]}",
        stage="silver",
        trigger=trigger,
        status="ok",
        duration_s=duration,
        data_fingerprint=panel_fingerprint,
        config_hash=settings.config_hash(),
    )

    return {
        "stage": "silver",
        "as_of": as_of,
        "status": "ok",
        "tickers": sorted(frames),
        "rows": int(len(panel)),
        "dq_failed": sum(1 for result in results if not result.passed),
        "quarantined": int(len(quarantine)),
        "fingerprints": fingerprints,
    }
