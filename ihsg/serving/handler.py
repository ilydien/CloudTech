"""Serving: handler murni yang dipakai bersama FastAPI dan Azure Function.

Sejak perbaikan konsistensi, handler **tidak lagi membaca champion**. Semua
angka yang disajikan berasal dari baris prediksi itu sendiri (``model_vol``,
``tau``, ``persistence``, ``ml_return_enabled``), sehingga mustahil terjadi
payload yang mencampur model baru dengan prediksi lama.

Handler hanya menyajikan ``as_of`` yang sudah punya penanda ``gold/published``,
jadi prediksi yang gagal separuh tidak akan pernah tampil sebagai "siap".
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from ihsg import __version__, dataset
from ihsg.config import Settings
from ihsg.storage import Sink, get_sink

SERVICE_NAME = "IHSG-Forecasting-Engine"
TICKER = "^JKSE"
REGIME_CLASSIFIER = "gru"


def _sink(settings: Settings, sink: Sink | None) -> Sink:
    return sink or get_sink(settings)


def latest_as_of(settings: Settings, sink: Sink | None = None) -> str | None:
    """Hanya ``as_of`` yang sudah dipublikasikan."""
    available = dataset.existing_as_of(_sink(settings, sink), dataset.GOLD_PUBLISHED)
    return available[-1] if available else None


def _metric_value(metrics: pd.DataFrame, model: str, metric: str, split: str, horizon: int = 1) -> float | None:
    rows = metrics[
        (metrics["model"] == model)
        & (metrics["metric"] == metric)
        & (metrics["split"] == split)
        & (metrics["horizon"] == horizon)
    ]
    if rows.empty:
        return None
    return float(rows["value"].iloc[0])


def _dq_summary(sink: Sink, as_of: str) -> dict:
    total = 0
    failed = 0
    for prefix in (dataset.SILVER_DQ, dataset.GOLD_DQ):
        frame = sink.try_read_parquet(dataset.as_of_key(prefix, as_of))
        if frame is None or frame.empty:
            continue
        total += int(len(frame))
        failed += int((~frame["passed"].astype(bool)).sum())
    return {"checks": total, "failed": failed}


def build_payload(
    settings: Settings,
    as_of: str | None = None,
    *,
    sink: Sink | None = None,
) -> dict:
    sink = _sink(settings, sink)
    as_of = as_of or latest_as_of(settings, sink)
    if as_of is None:
        raise LookupError("belum ada prediksi yang dipublikasikan; jalankan job predict dulu")

    published = sink.try_read_parquet(dataset.as_of_key(dataset.GOLD_PUBLISHED, as_of))
    if published is None or published.empty or not bool(published["ok"].iloc[0]):
        raise LookupError(f"prediksi {as_of} belum dipublikasikan")

    predictions = sink.read_parquet(dataset.as_of_key(dataset.GOLD_PREDICTIONS, as_of))
    params = sink.try_read_parquet(dataset.as_of_key(dataset.GOLD_PARAMS, as_of))
    metrics = sink.try_read_parquet(dataset.as_of_key(dataset.GOLD_METRICS, as_of))

    predictions = predictions.sort_values("horizon_step")
    first = predictions.iloc[0]
    last = predictions.iloc[-1]

    horizons = [
        {
            "horizon_step": int(row.horizon_step),
            "target_date": str(row.target_date),
            "forecast_variance_pct2": float(row.forecast_variance_pct2),
            "forecast_volatility_pct": float(row.forecast_volatility_pct),
            "cumulative_variance_pct2": float(row.cumulative_variance_pct2),
            "annualized_volatility_pct": float(row.annualized_volatility_pct),
            "risk_regime": str(row.risk_regime),
            "prob_high": float(row.prob_high),
            "expected_log_return_pct": float(row.expected_log_return_pct),
            "cumulative_expected_return_pct": float(row.cumulative_expected_return_pct),
            "median_close_price": float(row.median_close_price),
            "price_interval_95": [float(row.price_interval_low), float(row.price_interval_high)],
            "prob_up_vs_last_close": float(row.prob_up_vs_last_close),
            "direction_signal": str(row.direction_signal),
        }
        for row in predictions.itertuples(index=False)
    ]

    validation: dict[str, float | None] = {}
    if metrics is not None and not metrics.empty:
        model_vol = str(first.model_vol)
        validation = {
            "selected_model_qlike_val": _metric_value(metrics, model_vol, "qlike", "val"),
            "selected_model_qlike_test": _metric_value(metrics, model_vol, "qlike", "test"),
            "gru_qlike_test": _metric_value(metrics, "gru", "qlike", "test"),
            "har_qlike_test": _metric_value(metrics, "har", "qlike", "test"),
            "garch_qlike_test": _metric_value(metrics, "garch", "qlike", "test"),
            "ridge_r2_oos_val": _metric_value(metrics, "ridge", "r2_oos", "val"),
            "ridge_r2_oos_test": _metric_value(metrics, "ridge", "r2_oos", "test"),
        }

    payload = {
        "metadata": {
            "service": SERVICE_NAME,
            "version": __version__,
            "schema_version": 2,
            "ticker": TICKER,
            "as_of": as_of,
            "base_date": str(first.base_date),
            "last_close": float(first.last_close),
            "run_id": str(first.run_id),
            "data_fingerprint": str(params["data_fingerprint"].iloc[0]) if params is not None and not params.empty else "",
            "deterministic": True,
            "usage_note": (
                "Jalankan setelah penutupan pasar AS dan sebelum pembukaan BEI. "
                "Bukan saran investasi."
            ),
        },
        "model": {
            "volatility": str(first.model_vol),
            "return": str(first.model_ret),
            "regime_classifier": REGIME_CLASSIFIER,
            "ml_return_enabled": bool(first.ml_return_enabled),
            "tau_vol_pct": float(first.tau),
            "persistence": float(first.persistence),
            "window": int(params["window"].iloc[0]) if params is not None and not params.empty else None,
            "n_features": int(params["n_features"].iloc[0]) if params is not None and not params.empty else None,
        },
        "validation": validation,
        "forecast_horizons": horizons,
        "summary": {
            "horizon_days": int(len(horizons)),
            "cumulative_volatility_pct": float((last.cumulative_variance_pct2) ** 0.5),
            "cumulative_expected_return_pct": float(last.cumulative_expected_return_pct),
            "median_close_last_horizon": float(last.median_close_price),
            "regime_h1": str(first.risk_regime),
            "prob_high_h1": float(first.prob_high),
            "direction_signal_h1": str(first.direction_signal),
        },
        "data_quality": _dq_summary(sink, as_of),
    }
    if params is not None and not params.empty:
        payload["metadata"]["train_val_test_days"] = [
            int(params["n_train"].iloc[0]),
            int(params["n_val"].iloc[0]),
            int(params["n_test"].iloc[0]),
        ]
    return payload


def payload_text(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False)


def payload_sha256(payload: dict) -> str:
    return hashlib.sha256(payload_text(payload).encode("utf-8")).hexdigest()


def handle_request(
    settings: Settings,
    as_of: str | None = None,
    *,
    sink: Sink | None = None,
) -> tuple[int, dict]:
    """Kontrak untuk lapisan HTTP: ``(status_code, body)``."""
    try:
        payload = build_payload(settings, as_of, sink=sink)
    except LookupError as error:
        return 503, {"status": "unavailable", "error": str(error)}
    except Exception as error:  # noqa: BLE001
        return 500, {"status": "error", "error": f"{type(error).__name__}: {error}"}
    return 200, payload


def health(settings: Settings, *, sink: Sink | None = None) -> dict:
    try:
        as_of = latest_as_of(settings, sink)
    except Exception as error:  # noqa: BLE001
        return {"status": "degraded", "storage": "error", "error": str(error)}
    return {
        "status": "ok" if as_of else "empty",
        "storage": settings.sink,
        "latest_as_of": as_of,
        "version": __version__,
    }
