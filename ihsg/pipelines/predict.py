"""Gold/predict: ramalan 1..H hari memakai model champion.

Tidak ada pelatihan di sini. Semua parameter (bobot, scaler, tau, c_train,
long-run variance) dibaca dari artefak champion, sehingga prediksi untuk
``as_of`` tertentu deterministik.

Proyeksi varians memakai mean reversion GARCH:
``h_{t+k} = lrv + (h_{t+1} - lrv) * persistence^(k-1)``
(Bollerslev, 1986) dengan ``lrv`` = varians jangka panjang pada skala yang sama.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
from scipy import stats

from ihsg import dataset
from ihsg.config import Settings
from ihsg.models import seq
from ihsg.schemas import (
    GOLD_CHAMPION,
    GOLD_PREDICTIONS,
    GOLD_REGIME,
    validate_contract,
)
from ihsg.storage import Sink, get_sink


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
    sink: Sink | None = None,
) -> dict:
    settings.validate()
    sink = sink or get_sink(settings)
    as_of = as_of or settings.as_of or dataset.resolve_as_of(settings)

    frame = sink.read_parquet(dataset.as_of_key(dataset.GOLD_FEATURES, as_of))
    champion = _load_champion(sink)
    run_id = str(champion["run_id"])

    # --- pulihkan model & skala -------------------------------------------
    scaler = sink.read_parquet(dataset.artifact_key(run_id, "scaler.parquet"))
    feature_names = scaler["feature"].tolist()
    mean = scaler["mean"].to_numpy(dtype=float)
    scale = scaler["scale"].to_numpy(dtype=float)

    model, metadata = seq.loads(sink.read_bytes(dataset.artifact_key(run_id, "model.pt")))
    window = int(metadata["window"])
    if window != int(champion["window"]):
        raise RuntimeError(
            f"window artefak ({window}) tidak sama dengan champion ({champion['window']})"
        )
    if len(frame) < window + 1:
        raise RuntimeError(f"data fitur hanya {len(frame)} baris, butuh minimal {window + 1}")

    raw = frame[feature_names].to_numpy(dtype=float)
    scaled = (raw - mean) / scale
    window_input = scaled[-window:][None, :, :]

    variance_next, prob_high = seq.predict(model, window_input)
    h1 = float(np.clip(variance_next[0], 1e-6, None))
    p_high = float(prob_high[0])

    # --- parameter proyeksi -------------------------------------------------
    tau = float(champion["tau"])
    long_run = float(max(champion["long_run_var_pct2"], 1e-6))
    persistence = float(np.clip(champion["persistence"], 0.0, 0.9999))

    horizons = int(champion["horizons"])
    paths = [long_run + persistence ** (k - 1) * (h1 - long_run) for k in range(1, horizons + 1)]

    # --- komponen return ----------------------------------------------------
    n_real = len(frame) - 1
    last_close = float(frame["close"].iloc[n_real - 1])
    returns = frame["ret"].to_numpy(dtype=float)[:n_real]

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
        ret_next = float(np.nanmean(returns)) if len(returns) else 0.0
        model_ret = "expanding_mean"

    mean_return = float(np.nanmean(returns)) if len(returns) else 0.0

    # --- bangun 7 baris -----------------------------------------------------
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
    for step, (target_date, path_variance) in enumerate(zip(dates, paths), start=1):
        mu = ret_next if step == 1 else mean_return
        cumulative_variance += path_variance
        cumulative_return += mu
        cumulative_sd = float(np.sqrt(cumulative_variance))
        volatility = float(np.sqrt(path_variance))

        rows.append({
            "as_of": as_of,
            "run_id": run_id,
            "model_vol": "gru",
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
            "prob_high": p_high,
            "expected_log_return_pct": mu,
            "cumulative_expected_return_pct": cumulative_return,
            "median_close_price": last_close * float(np.exp(cumulative_return / 100.0)),
            "price_interval_low": last_close * float(np.exp((cumulative_return - 1.96 * cumulative_sd) / 100.0)),
            "price_interval_high": last_close * float(np.exp((cumulative_return + 1.96 * cumulative_sd) / 100.0)),
            "prob_up_vs_last_close": float(stats.norm.cdf(cumulative_return / cumulative_sd)) if cumulative_sd > 0 else 0.5,
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
        "prob_high": p_high,
        "risk_regime": predictions["risk_regime"],
    })
    validate_contract(regime, GOLD_REGIME)
    sink.write_parquet(regime, dataset.as_of_key(dataset.GOLD_REGIME, as_of), overwrite=True)

    return {
        "stage": "predict",
        "as_of": as_of,
        "run_id": run_id,
        "status": "ok",
        "base_date": base_date,
        "last_close": round(last_close, 4),
        "volatility_h1_pct": round(float(np.sqrt(h1)), 4),
        "prob_high": round(p_high, 4),
        "regime_h1": rows[0]["risk_regime"],
        "direction_signal": rows[0]["direction_signal"],
        "median_close_7d": round(float(rows[-1]["median_close_price"]), 4),
    }
