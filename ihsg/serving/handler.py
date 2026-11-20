"""Serving: handler murni yang dipakai bersama FastAPI dan Azure Function.

Aturan desain:
* handler **tidak** melatih model dan **tidak** mengunduh data. Ia hanya
  membaca tabel gold yang sudah dihitung oleh job ``predict``.
* keluaran berupa ``dict`` siap-``json.dumps``. JSON hanya ada di sini sebagai
  *response HTTP*, bukan media penyimpanan.
* payload deterministik: tanpa timestamp dinding, sehingga dua permintaan pada
  ``as_of`` yang sama menghasilkan byte yang identik.
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


def _sink(settings: Settings, sink: Sink | None) -> Sink:
    return sink or get_sink(settings)


def latest_as_of(settings: Settings, sink: Sink | None = None) -> str | None:
    available = dataset.existing_as_of(_sink(settings, sink), dataset.GOLD_PREDICTIONS)
    return available[-1] if available else None


def _metric_value(metrics: pd.DataFrame, model: str, metric: str, split: str) -> float | None:
    rows = metrics[
        (metrics["model"] == model)
        & (metrics["metric"] == metric)
        & (metrics["split"] == split)
        & (metrics["horizon"] == 1)
    ]
    if rows.empty:
        return None
    return float(rows["value"].iloc[0])


def _dq_summary(sink: Sink, as_of: str) -> dict:
    frame = sink.try_read_parquet(dataset.as_of_key(dataset.SILVER_DQ, as_of))
    if frame is None or frame.empty:
        return {"checks": 0, "failed": 0}
    return {
        "checks": int(len(frame)),
        "failed": int((~frame["passed"].astype(bool)).sum()),
        "soft_failed": int(((~frame["passed"].astype(bool)) & (frame["severity"] != "hard")).sum()),
    }


def build_payload(
    settings: Settings,
    as_of: str | None = None,
    *,
    sink: Sink | None = None,
) -> dict:
    sink = _sink(settings, sink)
    as_of = as_of or latest_as_of(settings, sink)
    if as_of is None:
        raise LookupError("belum ada gold/predictions; jalankan job predict dulu")

    predictions = sink.read_parquet(dataset.as_of_key(dataset.GOLD_PREDICTIONS, as_of))
    champion_frame = sink.try_read_parquet(dataset.GOLD_CHAMPION)
    if champion_frame is None or champion_frame.empty:
        raise LookupError("champion belum ada; jalankan job train terlebih dahulu")
    champion = champion_frame.iloc[0]
    metrics = sink.try_read_parquet(dataset.as_of_key(dataset.GOLD_METRICS, as_of))
    params = sink.try_read_parquet(dataset.as_of_key(dataset.GOLD_PARAMS, as_of))

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
        validation = {
            "gru_qlike_val": _metric_value(metrics, "gru", "qlike", "val"),
            "gru_qlike_test": _metric_value(metrics, "gru", "qlike", "test"),
            "har_qlike_test": _metric_value(metrics, "har", "qlike", "test"),
            "garch_qlike_test": _metric_value(metrics, "garch", "qlike", "test"),
            "gru_rmse_vol_test": _metric_value(metrics, "gru", "rmse_vol", "test"),
            "ridge_r2_oos_test": _metric_value(metrics, "ridge", "r2_oos", "test"),
        }

    payload = {
        "metadata": {
            "service": SERVICE_NAME,
            "version": __version__,
            "schema_version": 1,
            "ticker": TICKER,
            "as_of": as_of,
            "base_date": str(first.base_date),
            "last_close": float(first.last_close),
            "run_id": str(first.run_id),
            "data_fingerprint": str(champion["data_fingerprint"]),
            "config_hash": str(champion["config_hash"]),
            "deterministic": True,
            "usage_note": (
                "Jalankan setelah penutupan pasar AS dan sebelum pembukaan BEI. "
                "Bukan saran investasi."
            ),
        },
        "model": {
            "volatility": str(champion["model_vol"]),
            "return": str(champion["model_ret"]),
            "ml_return_enabled": bool(champion["ml_return_enabled"]),
            "window": int(champion["window"]),
            "n_features": int(champion["n_features"]),
            "tau_vol_pct": float(champion["tau"]),
            "c_train": float(champion["c_train"]),
            "persistence": float(champion["persistence"]),
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
