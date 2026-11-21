"""Bronze: unduh data mentah dan simpan sebagai partisi Parquet append-only.

Sifat bronze:
* **append-only** - partisi ``ingest_date=<as_of>`` tidak pernah ditimpa.
  Partisi yang sudah ada di-skip (kecuali ``force=True``).
* **incremental** - baris yang sudah ada tidak diunduh ulang. Rentang unduhan
  dimulai dari tanggal terakhir yang sudah tersimpan, sehingga storage tumbuh
  linear, bukan kuadratik.
* **tanpa transformasi nilai** - hanya konversi wadah ke Parquet (diwajibkan
  oleh aturan "non CSV").
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from ihsg import dataset
from ihsg.config import Settings
from ihsg.instruments import INDEX_KEY, get_instruments
from ihsg.providers import download_raw, provider_version
from ihsg.quality import (
    HARD,
    SOFT,
    QualityError,
    check_freshness,
    check_not_null,
    check_positive,
)
from ihsg.schemas import (
    BRONZE_MANIFEST,
    BRONZE_PRICES,
    fingerprint,
    validate_contract,
)
from ihsg.storage import Sink, get_sink


def _latest_bronze_date(sink: Sink, ticker_key: str) -> str | None:
    prefix = f"{dataset.BRONZE_PRICES}/ticker={ticker_key}"
    ingest_dates = dataset.existing_ingest_dates(sink, prefix)
    if not ingest_dates:
        return None
    newest = max(ingest_dates)
    key = dataset.bronze_prices_key(ticker_key, newest)
    frame = sink.read_parquet(key)
    if frame.empty:
        return None
    return pd.Timestamp(frame["Date"].max()).date().isoformat()


def run(
    settings: Settings,
    as_of: str | None = None,
    *,
    force: bool = False,
    sink: Sink | None = None,
) -> dict:
    settings.validate()
    sink = sink or get_sink(settings)
    candidate_as_of = as_of or settings.as_of or dataset.resolve_as_of(settings)
    dataset.assert_trading_day(candidate_as_of, settings.idx_holidays)

    instruments = get_instruments(settings)
    last_dates = {item.key: _latest_bronze_date(sink, item.key) for item in instruments}

    pending = [
        item.key
        for item in instruments
        if force or last_dates[item.key] is None or last_dates[item.key] < candidate_as_of
    ]
    if not pending:
        known = [value for value in last_dates.values() if value]
        # Bronze hanya berisi sesi nyata, jadi tanggal terakhirnya adalah sesi
        # terakhir yang valid <= candidate (mis. saat candidate jatuh di libur).
        return {
            "stage": "ingest",
            "as_of": min(candidate_as_of, max(known)) if known else candidate_as_of,
            "candidate_as_of": candidate_as_of,
            "status": "up_to_date",
            "written": [],
            "skipped": [item.key for item in instruments],
        }

    # Satu panggilan API untuk semua ticker; titik mulai = yang paling tertinggal.
    starts = []
    for item in instruments:
        if item.key not in pending:
            continue
        last = last_dates[item.key]
        starts.append(settings.start_date if last is None else (date.fromisoformat(last) + timedelta(days=1)).isoformat())
    request_start = min(starts) if starts else settings.start_date

    batches = download_raw(instruments, request_start, candidate_as_of)
    version = provider_version()

    if not batches:
        known = [value for value in last_dates.values() if value]
        if not known:
            raise RuntimeError(
                "unduhan kosong pada run pertama: periksa koneksi/provider atau START_DATE"
            )
        # Provider belum menerbitkan sesi baru. Jangan gagal, jangan tulis apa pun:
        # pakai tanggal data terakhir yang sudah ada agar seluruh pipeline tetap
        # memakai as_of yang konsisten.
        return {
            "stage": "ingest",
            "as_of": max(known),
            "candidate_as_of": candidate_as_of,
            "status": "up_to_date",
            "note": "provider belum menerbitkan data baru",
            "written": [],
            "skipped": [item.key for item in instruments],
        }

    # Clamp as_of ke data yang benar-benar tersedia untuk IHSG.
    #
    # Provider sering tertinggal 1-2 sesi (atau menutup lebih awal). Kalau kita
    # memaksa as_of = hari bursa terakhir menurut kalender, fitur hari-depan
    # akan kosong dan pipeline gagal - atau lebih buruk, menghasilkan ramalan
    # yang tidak sejajar dengan data. Titik acuan yang benar adalah sesi
    # terakhir yang datanya ada.
    index_last = next(
        (
            pd.Timestamp(batch.prices["Date"].max()).date().isoformat()
            for batch in batches
            if batch.instrument.key == INDEX_KEY and not batch.prices.empty
        ),
        None,
    )
    as_of = candidate_as_of
    if index_last is not None and index_last < candidate_as_of:
        as_of = index_last

    written: list[str] = []
    skipped: list[str] = []
    manifest_rows: list[dict] = []

    for batch in batches:
        item = batch.instrument
        last = last_dates[item.key]
        prices = batch.prices
        if last is not None:
            prices = prices[prices["Date"] > pd.Timestamp(last)].reset_index(drop=True)
        actions = batch.actions

        if prices.empty:
            skipped.append(item.key)
        else:
            validate_contract(prices, BRONZE_PRICES)
            check_not_null(prices, "bronze.prices", "bronze", ["Date", "Close"]).raise_if_hard()
            # Positivitas hanya wajib untuk IHSG (target model). Futures seperti
            # CL=F boleh negatif - WTI pernah ditutup -37,63 pada April 2020.
            check_positive(
                prices, "bronze.prices", "bronze", "Close",
                severity=SOFT if item.key != INDEX_KEY else HARD,
            ).raise_if_hard()

            key = dataset.bronze_prices_key(item.key, as_of)
            sink.write_parquet(prices, key, overwrite=force)
            sink.write_parquet(
                actions[actions["Date"] >= prices["Date"].min()].reset_index(drop=True),
                dataset.bronze_actions_key(item.key, as_of),
                overwrite=force,
            )
            written.append(item.key)

        manifest_rows.append(
            {
                "ingest_date": as_of,
                "ticker": item.ticker,
                "ticker_key": item.key,
                "rows": int(len(prices)),
                "date_min": pd.Timestamp(prices["Date"].min()).date().isoformat() if not prices.empty else "",
                "date_max": pd.Timestamp(prices["Date"].max()).date().isoformat() if not prices.empty else "",
                "fingerprint": fingerprint(prices),
                "provider": "yfinance",
                "provider_version": version,
                "request_start": request_start,
                "request_end": (date.fromisoformat(candidate_as_of) + timedelta(days=1)).isoformat(),
            }
        )

    if manifest_rows:
        manifest = pd.DataFrame(manifest_rows)
        validate_contract(manifest, BRONZE_MANIFEST)
        sink.write_parquet(manifest, dataset.bronze_manifest_key(as_of), overwrite=True)

    # Guard freshness hanya untuk instrumen acuan (IHSG). Pasar lain boleh
    # tertinggal karena beda kalender/libur.
    index_dates = dataset.existing_ingest_dates(sink, f"{dataset.BRONZE_PRICES}/ticker={INDEX_KEY}")
    if not index_dates:
        raise QualityError("bronze IHSG kosong setelah ingest; periksa provider/log")
    index_frame = sink.read_parquet(dataset.bronze_prices_key(INDEX_KEY, max(index_dates)))
    freshness = check_freshness(index_frame, "bronze.prices", "bronze", as_of, column="Date")
    if not freshness.passed:
        raise QualityError(
            f"data IHSG belum mencapai {as_of} (terakhir {freshness.details.get('latest')}). "
            "Kemungkinan provider terlambat atau as_of terlalu maju."
        )

    return {
        "stage": "ingest",
        "as_of": as_of,
        "candidate_as_of": candidate_as_of,
        "status": "ok",
        "request_start": request_start,
        "written": written,
        "skipped": skipped,
        "provider_version": version,
    }
