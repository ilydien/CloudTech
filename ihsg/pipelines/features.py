"""Gold/features: matriks fitur ber-lag + target IHSG.

Aturan anti-kebocoran:
* semua fitur memakai informasi sampai ``t-1`` (``shift(1)`` eksplisit),
* baris tambahan untuk **hari bursa berikutnya** dibuat dengan indeks tanggal
  yang diperpanjang, sehingga fitur hari-depan benar-benar tersedia,
* ``c_train`` (skala proxy Parkinson) dan ``tau`` (ambang rezim) dihitung
  HANYA dari data latih.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ihsg import dataset, indicators
from ihsg.config import Settings
from ihsg.instruments import INDEX_KEY, get_instruments
from ihsg.quality import HARD, SOFT, CheckResult, write_report
from ihsg.schemas import GOLD_FEATURES, GOLD_PARAMS, fingerprint, validate_contract
from ihsg.storage import Sink, get_sink

TARGET_COLUMNS = ("ret", "park", "close", "prev_close")
EPS = 1e-4


def build_frame(panel: pd.DataFrame, settings: Settings, as_of: str) -> tuple[pd.DataFrame, dict]:
    """Bangun fitur + target. Baris terakhir = hari bursa berikutnya."""
    if panel.empty:
        raise ValueError("panel silver kosong")

    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(panel["date"]).unique()))
    last_session = pd.Timestamp(sessions[-1]).date().isoformat()
    if last_session != as_of:
        raise ValueError(
            f"silver berakhir {last_session}, bukan {as_of}. "
            "Jalankan ingest untuk tanggal itu terlebih dahulu."
        )

    target_day = dataset.next_trading_day(as_of, settings.idx_holidays)
    idx = sessions.append(pd.DatetimeIndex([pd.Timestamp(target_day)]))

    frame = panel.set_index("date").reindex(idx)

    features: dict[str, pd.Series] = {}
    skipped_volume: list[str] = []

    for instrument in get_instruments(settings):
        key = instrument.key
        adj = frame[f"{key}_adj_close"].astype(float)
        high = frame[f"{key}_high"].astype(float)
        low = frame[f"{key}_low"].astype(float)
        volume = frame[f"{key}_volume"].astype(float)

        returns = indicators.log_return_pct(adj)
        park = indicators.parkinson_var(high, low)

        features[f"{key}_ret_lag1"] = returns.shift(1)
        features[f"{key}_park_var_lag1"] = park.shift(1)
        features[f"{key}_ret_ma5_lag1"] = returns.rolling(5).mean().shift(1)
        features[f"{key}_rv5_lag1"] = (returns**2).rolling(5).mean().shift(1)
        if float((volume.fillna(0) > 0).mean()) > 0.9:
            features[f"{key}_vol_chg_lag1"] = np.log((volume + 1.0) / (volume.shift(1) + 1.0)).shift(1)
        else:
            skipped_volume.append(key)

    index_ret = features[f"{INDEX_KEY}_ret_lag1"]
    index_park = features[f"{INDEX_KEY}_park_var_lag1"]
    features[f"{INDEX_KEY}_sq_ret_lag1"] = index_ret**2
    features[f"{INDEX_KEY}_rv22_lag1"] = (index_ret**2).rolling(22).mean()
    features[f"{INDEX_KEY}_park_ma5_lag1"] = index_park.rolling(5).mean()
    features[f"{INDEX_KEY}_park_ma22_lag1"] = index_park.rolling(22).mean()

    out = pd.DataFrame(features)
    index_close = frame[f"{INDEX_KEY}_close"].astype(float)
    index_adj = frame[f"{INDEX_KEY}_adj_close"].astype(float)
    out["ret"] = indicators.log_return_pct(index_adj)
    out["park"] = indicators.parkinson_var(frame[f"{INDEX_KEY}_high"], frame[f"{INDEX_KEY}_low"])
    out["close"] = index_close
    out["prev_close"] = index_close.shift(1)

    feature_columns = [column for column in out.columns if column not in TARGET_COLUMNS]
    complete = out[feature_columns].notna().all(axis=1)
    if not complete.any():
        raise ValueError("tidak ada baris dengan fitur lengkap; data terlalu pendek")

    # Buang SEMUA baris yang fiturnya tidak lengkap, bukan hanya baris awal.
    # Kalau tidak, satu gap panjang pada salah satu instrumen (libur > FFILL_LIMIT)
    # akan menghasilkan NaN yang merambat ke standardisasi dan training.
    total_rows = len(out)
    out = out.loc[complete].copy()
    dropped = int(total_rows - len(out))
    out.index.name = "date"
    out = out.reset_index()

    # Baris terakhir harus hari bursa berikutnya, dan fiturnya wajib lengkap.
    if pd.Timestamp(out["date"].max()) != pd.Timestamp(target_day):
        raise AssertionError(
            f"fitur hari-depan ({target_day}) tidak lengkap; "
            f"baris lengkap terakhir = {pd.Timestamp(out['date'].max()).date()}"
        )
    if out[list(TARGET_COLUMNS)].iloc[:-1].isna().any().any():
        raise AssertionError("target historis mengandung NaN setelah pemangkasan")

    meta = {"feature_columns": feature_columns, "skipped_volume": skipped_volume, "target_day": target_day, "rows_dropped": dropped}
    return out, meta


def compute_params(frame: pd.DataFrame, settings: Settings, as_of: str) -> dict:
    """Parameter split & skala. Hanya memakai data latih untuk ``c_train``/``tau``."""
    n_real = len(frame) - 1
    i_tr = int(n_real * settings.train_frac)
    i_va = int(n_real * (settings.train_frac + settings.val_frac))

    ret = frame["ret"].to_numpy(dtype=float)
    park = frame["park"].to_numpy(dtype=float)

    mean_sq = float(np.nanmean(ret[:i_tr] ** 2))
    mean_park = float(np.nanmean(park[:i_tr]))
    c_train = mean_sq / mean_park if mean_park > 0 else 1.0

    tv = c_train * park
    tau = float(np.nanquantile(np.sqrt(np.maximum(tv[:i_tr], EPS)), settings.tau_quantile))
    long_run_var = float(np.nanmean(tv[:i_tr]))

    return {
        "as_of": as_of,
        "c_train": c_train,
        "tau": tau,
        "long_run_var_pct2": long_run_var,
        "persistence": float(settings.ewma_lambda),
        "n_train": int(i_tr),
        "n_val": int(i_va - i_tr),
        "n_test": int(n_real - i_va),
        "i_tr": int(i_tr),
        "i_va": int(i_va),
        "n_real": int(n_real),
        "n_features": int(len([c for c in frame.columns if c not in TARGET_COLUMNS and c != "date"])),
        "window": int(settings.window),
        "data_fingerprint": fingerprint(frame),
    }


def run(
    settings: Settings,
    as_of: str | None = None,
    *,
    sink: Sink | None = None,
) -> dict:
    settings.validate()
    sink = sink or get_sink(settings)
    as_of = as_of or settings.as_of or dataset.resolve_as_of(settings)

    panel = sink.read_parquet(f"{dataset.SILVER_PANEL}/{dataset.PART_FILE}")
    frame, meta = build_frame(panel, settings, as_of)
    validate_contract(frame, GOLD_FEATURES)

    params = compute_params(frame, settings, as_of)
    params_frame = pd.DataFrame([params])
    validate_contract(params_frame, GOLD_PARAMS)

    sink.write_parquet(frame, dataset.as_of_key(dataset.GOLD_FEATURES, as_of), overwrite=True)
    sink.write_parquet(params_frame, dataset.as_of_key(dataset.GOLD_PARAMS, as_of), overwrite=True)

    # Kualitas data layer gold: kelengkapan fitur & target, dan berapa baris
    # yang dibuang karena warm-up/gap. Dicatat, bukan menghentikan run.
    results = [
        CheckResult(
            layer="gold",
            table="gold.features",
            check_name="feature_completeness",
            severity=HARD,
            failed_count=int(frame[meta["feature_columns"]].isna().any(axis=1).sum()),
            total_count=len(frame),
            details={"columns": len(meta["feature_columns"])},
        ),
        CheckResult(
            layer="gold",
            table="gold.features",
            check_name="target_completeness",
            severity=HARD,
            failed_count=int(frame["ret"].iloc[:-1].isna().sum()),
            total_count=max(len(frame) - 1, 1),
            details={"excluded": "baris hari-depan"},
        ),
        CheckResult(
            layer="gold",
            table="gold.features",
            check_name="rows_dropped_incomplete",
            severity=SOFT,
            failed_count=int(meta["rows_dropped"]),
            total_count=len(frame) + int(meta["rows_dropped"]),
            details={"reason": "warmup/ffill limit"},
        ),
    ]
    for result in results:
        result.raise_if_hard()
    write_report(sink, results, as_of, layer="gold")

    return {
        "stage": "features",
        "as_of": as_of,
        "status": "ok",
        "rows": int(len(frame)),
        "features": len(meta["feature_columns"]),
        "rows_dropped": int(meta["rows_dropped"]),
        "skipped_volume": meta["skipped_volume"],
        "target_day": meta["target_day"],
        "c_train": round(params["c_train"], 6),
        "tau": round(params["tau"], 6),
    }
