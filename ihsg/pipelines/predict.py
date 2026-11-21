"""Gold/predict: ramalan 1..H hari dari kandidat pemenang.

Sejak pemilihan model diperbaiki, ``predict`` **tidak lagi** menjalankan GRU.
Ia membaca ``gold/candidates`` (ramalan h=1 semua model + parameter mean
reversion) dan memakai baris milik model champion. Konsekuensinya:

* prediksi tidak bisa menyimpang dari hasil evaluasi (tidak ada train/serve skew),
* prediksi cepat dan tidak butuh torch,
* ramalan tetap deterministik untuk ``as_of`` yang sama.

Proyeksi varians: ``h_{t+k} = lrv + persistence^(k-1) * (h1 - lrv)``
(Bollerslev, 1986). Model flat (historis, EWMA) diproyeksikan datar.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
from scipy import stats

from ihsg import dataset
from ihsg.config import Settings
from ihsg.runlog import append_run
from ihsg.schemas import (
    GOLD_CANDIDATES,
    GOLD_CHAMPION,
    GOLD_PREDICTIONS,
    GOLD_PUBLISHED,
    GOLD_REGIME,
    validate_contract,
)
from ihsg.storage import Sink, get_sink

FLAT_MODELS = ("hist", "ewma")


def _load_champion(sink: Sink) -> pd.Series:
    frame = sink.try_read_parquet(dataset.GOLD_CHAMPION)
    if frame is None or frame.empty:
        raise RuntimeError(
            "champion belum ada: jalankan `python -m ihsg.cli train` sebelum predict"
        )
    validate_contract(frame, GOLD_CHAMPION)
    return frame.iloc[0]


def run(
    settings: Settings,
    as_of: str | None = None,
    *,
    trigger: str = "schedule",
    sink: Sink | None = None,
) -> dict:
    settings.validate()
    started = time.perf_counter()
    sink = sink or get_sink(settings)
    as_of = as_of or settings.as_of or dataset.resolve_as_of(settings)

    frame = sink.read_parquet(dataset.as_of_key(dataset.GOLD_FEATURES, as_of))
    params = sink.read_parquet(dataset.as_of_key(dataset.GOLD_PARAMS, as_of)).iloc[0]
    champion = _load_champion(sink)

    candidates = sink.read_parquet(dataset.as_of_key(dataset.GOLD_CANDIDATES, as_of))
    validate_contract(candidates, GOLD_CANDIDATES)

    model_vol = str(champion["model_vol"])
    selected = candidates[candidates["model"] == model_vol]
    if selected.empty:
        # Champion lama yang modelnya tidak tersedia pada run ini (mis. GARCH
        # gagal konvergen). Pakai kandidat terpilih agar prediksi tetap tersedia,
        # dan laporkan penggantinya.
        selected = candidates[candidates["selected"]]
        model_vol = str(selected["model"].iloc[0])
    row = selected.iloc[0]

    h1 = float(np.clip(row["h1_var"], 1e-6, None))
    uncond = float(np.clip(row["uncond"], 1e-6, None))
    persistence = float(np.clip(row["persistence"], 0.0, 0.9999))
    run_id = str(row["run_id"])

    gru_rows = candidates[candidates["model"] == "gru"]
    prob_high = (
        float(gru_rows["prob_high"].iloc[0])
        if not gru_rows.empty and np.isfinite(gru_rows["prob_high"].iloc[0])
        else float("nan")
    )

    tau = float(params["tau"])
    horizons = int(champion["horizons"])

    if model_vol in FLAT_MODELS:
        paths = [h1] * horizons
    else:
        paths = [uncond + persistence ** (k - 1) * (h1 - uncond) for k in range(1, horizons + 1)]

    # --- komponen return ----------------------------------------------------
    n_real = len(frame) - 1
    last_close = float(frame["close"].iloc[n_real - 1])
    returns = frame["ret"].to_numpy(dtype=float)[:n_real]
    mean_return = float(np.nanmean(returns)) if len(returns) else 0.0

    ml_return_enabled = bool(champion["ml_return_enabled"])
    if ml_return_enabled:
        import joblib

        blob = joblib.load(io.BytesIO(sink.read_bytes(dataset.artifact_key(run_id, "ridge.joblib"))))
        ridge = blob["model"]
        ridge_features = blob["features"]
        latest_row = frame[ridge_features].to_numpy(dtype=float)[-1:].reshape(1, -1)
        ret_next = float(ridge.predict(latest_row)[0])
        model_ret = "ridge"
    else:
        ret_next = mean_return
        model_ret = "expanding_mean"

    # --- 7 baris ------------------------------------------------------------
    base_date = pd.Timestamp(frame["date"].iloc[n_real - 1]).date().isoformat()
    dates = pd.bdate_range(
        start=(pd.Timestamp(base_date) + pd.Timedelta(days=1)).date().isoformat(),
        periods=horizons,
        freq="C",
        holidays=list(settings.idx_holidays),
    )

    rows: list[dict] = []
    cumulative_variance = 0.0
    cumulative_return = 0.0
    for step, (target_date, path_variance) in enumerate(zip(dates, paths, strict=False), start=1):
        mu = ret_next if step == 1 else mean_return
        cumulative_variance += path_variance
        cumulative_return += mu
        cumulative_sd = float(np.sqrt(cumulative_variance))
        volatility = float(np.sqrt(path_variance))

        rows.append({
            "as_of": as_of,
            "run_id": run_id,
            "model_vol": model_vol,
            "model_ret": model_ret,
            "base_date": base_date,
            "last_close": last_close,
            "horizon_step": step,
            "target_date": target_date.date().isoformat(),
            "forecast_variance_pct2": path_variance,
            "forecast_volatility_pct": volatility,
            "cumulative_variance_pct2": cumulative_variance,
            "annualized_volatility_pct": volatility * float(np.sqrt(252.0)),
            "risk_regime": "HIGH" if volatility > tau else "NORMAL",
            "prob_high": prob_high,
            "tau": tau,
            "persistence": persistence,
            "ml_return_enabled": ml_return_enabled,
            "expected_log_return_pct": mu,
            "cumulative_expected_return_pct": cumulative_return,
            "median_close_price": last_close * float(np.exp(cumulative_return / 100.0)),
            "price_interval_low": last_close * float(
                np.exp((cumulative_return - 1.96 * cumulative_sd) / 100.0)
            ),
            "price_interval_high": last_close * float(
                np.exp((cumulative_return + 1.96 * cumulative_sd) / 100.0)
            ),
            "prob_up_vs_last_close": (
                float(stats.norm.cdf(cumulative_return / cumulative_sd))
                if cumulative_sd > 0 else 0.5
            ),
            "direction_signal": (
                ("UP" if ret_next >= 0 else "DOWN") if (step == 1 and ml_return_enabled) else "NO_SIGNAL"
            ),
        })

    predictions = pd.DataFrame(rows)
    validate_contract(predictions, GOLD_PREDICTIONS)
    sink.write_parquet(predictions, dataset.as_of_key(dataset.GOLD_PREDICTIONS, as_of), overwrite=True)

    regime = pd.DataFrame({
        "as_of": as_of,
        "run_id": run_id,
        "horizon_step": predictions["horizon_step"],
        "target_date": predictions["target_date"],
        "prob_high": prob_high,
        "risk_regime": predictions["risk_regime"],
    })
    validate_contract(regime, GOLD_REGIME)
    sink.write_parquet(regime, dataset.as_of_key(dataset.GOLD_REGIME, as_of), overwrite=True)

    published = pd.DataFrame([{
        "as_of": as_of,
        "run_id": run_id,
        "model_vol": model_vol,
        "horizons": int(len(predictions)),
        "ok": True,
    }])
    validate_contract(published, GOLD_PUBLISHED)
    sink.write_parquet(published, dataset.as_of_key(dataset.GOLD_PUBLISHED, as_of), overwrite=True)

    append_run(
        sink,
        as_of=as_of,
        run_id=run_id,
        stage="predict",
        trigger=trigger,
        status="ok",
        duration_s=time.perf_counter() - started,
        data_fingerprint=str(params["data_fingerprint"]),
        config_hash=settings.config_hash(),
    )

    return {
        "stage": "predict",
        "as_of": as_of,
        "run_id": run_id,
        "status": "ok",
        "model_vol": model_vol,
        "model_ret": model_ret,
        "base_date": base_date,
        "last_close": round(last_close, 4),
        "volatility_h1_pct": round(float(np.sqrt(h1)), 4),
        "prob_high": None if not np.isfinite(prob_high) else round(prob_high, 4),
        "regime_h1": rows[0]["risk_regime"],
        "direction_signal": rows[0]["direction_signal"],
        "median_close_7d": round(float(rows[-1]["median_close_price"]), 4),
    }
