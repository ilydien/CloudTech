"""Gold/train: latih GRU, bandingkan SEMUA kandidat, dan pilih pemenang.

Dua perubahan penting dibanding versi pertama:

1. **Pemilihan model jujur.** Baseline (HAR, GARCH, EWMA, historis) ikut menjadi
   kandidat, bukan sekadar pembanding di tabel. Champion = QLIKE validasi
   terendah di antara semua kandidat. Sebelumnya GRU selalu dipromosikan walau
   kalah, karena gate hanya membandingkannya dengan champion sebelumnya.

2. **Evaluasi multi-horizon.** QLIKE/RMSE varians kumulatif untuk h=1..7 plus
   uji Diebold-Mariano per horizon, supaya ramalan 7 hari punya bukti.

Ramalan h=1 tiap kandidat untuk sesi berikutnya disimpan di ``gold/candidates``.
``predict`` hanya membaca tabel itu, sehingga tidak ada lagi train/serve skew
dan prediksi tidak membutuhkan torch.
"""

from __future__ import annotations

import hashlib
import io
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ihsg import dataset, eval as metrics
from ihsg.config import JAKARTA_TZ, Settings
from ihsg.models import baselines, seq
from ihsg.pipelines.features import TARGET_COLUMNS
from ihsg.runlog import append_run, git_sha
from ihsg import tracking
from ihsg.schemas import (
    GOLD_CANDIDATES,
    GOLD_CHAMPION,
    GOLD_METRICS,
    GOLD_REGIME_EVAL,
    GOLD_VOL_EVAL,
    fingerprint,
    validate_contract,
)
from ihsg.storage import Sink, get_sink

EPS = 1e-6
FLAT_MODELS = ("hist", "ewma")


def make_run_id(as_of: str, config_hash: str, data_fingerprint: str) -> str:
    payload = f"{as_of}|{config_hash}|{data_fingerprint}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _standardise(train: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0, ddof=0)
    scale[scale < 1e-9] = 1.0
    return (data - mean) / scale, mean, scale


# ---------------------------------------------------------------------------
# Baris metrik
# ---------------------------------------------------------------------------
def _metric_rows(
    ends: np.ndarray,
    actual: np.ndarray,
    forecasts: dict[str, np.ndarray],
    *,
    as_of: str,
    run_id: str,
    split: str,
) -> list[dict]:
    rows: list[dict] = []
    for name, series in forecasts.items():
        values = series[ends]
        valid = np.isfinite(actual[ends]) & np.isfinite(values)
        if valid.sum() < 10:
            continue
        actual_valid = actual[ends][valid]
        forecast_valid = values[valid]
        rows.append(
            {
                "as_of": as_of, "run_id": run_id, "split": split, "model": name,
                "metric": "qlike", "horizon": 1,
                "value": float(metrics.qlike(actual_valid, forecast_valid).mean()),
            }
        )
        rows.append(
            {
                "as_of": as_of, "run_id": run_id, "split": split, "model": name,
                "metric": "rmse_vol", "horizon": 1,
                "value": float(
                    np.sqrt(np.mean((np.sqrt(actual_valid) - np.sqrt(forecast_valid)) ** 2))
                ),
            }
        )
    return rows


def _dm_rows(
    ends: np.ndarray,
    actual: np.ndarray,
    forecasts: dict[str, np.ndarray],
    *,
    as_of: str,
    run_id: str,
    split: str,
    challenger: str = "gru",
    benches: tuple[str, ...] = ("har", "garch", "ewma", "hist"),
) -> list[dict]:
    rows: list[dict] = []
    reference = forecasts.get(challenger)
    if reference is None:
        return rows

    for bench in benches:
        other = forecasts.get(bench)
        if other is None:
            continue
        valid = (
            np.isfinite(actual[ends])
            & np.isfinite(reference[ends])
            & np.isfinite(other[ends])
        )
        if int(valid.sum()) < 30:
            continue
        actual_valid = actual[ends][valid]
        statistic, p_value = metrics.dm_test(
            metrics.qlike(actual_valid, reference[ends][valid]),
            metrics.qlike(actual_valid, other[ends][valid]),
        )
        rows.append(
            {
                "as_of": as_of, "run_id": run_id, "split": split, "model": challenger,
                "metric": f"dm_p_vs_{bench}", "horizon": 1, "value": float(p_value),
            }
        )
        rows.append(
            {
                "as_of": as_of, "run_id": run_id, "split": split, "model": challenger,
                "metric": f"dm_stat_vs_{bench}", "horizon": 1, "value": float(statistic),
            }
        )
    return rows


def _cumulative_forecast(
    model: str, v1: np.ndarray, uncond: np.ndarray, persistence: np.ndarray, h: int
) -> np.ndarray:
    if model in FLAT_MODELS:
        return h * v1
    per = np.clip(persistence, 0.0, 0.9999)
    factor = (1.0 - per**h) / (1.0 - per)
    return h * uncond + (v1 - uncond) * factor


def _multi_horizon_rows(
    ends: np.ndarray,
    tv: np.ndarray,
    forecasts: dict[str, np.ndarray],
    aux: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    as_of: str,
    run_id: str,
    split: str,
    n_real: int,
    horizons: int,
) -> list[dict]:
    """QLIKE/RMSE varians kumulatif h=2..H + DM tiap kandidat vs HAR.

    h=1 tidak diulang di sini: sudah ditulis oleh ``_metric_rows``/``_dm_rows``
    dengan kunci (model, metric, horizon) yang sama.
    """
    rows: list[dict] = []
    for h in range(2, horizons + 1):
        horizon_mask = ends <= (n_real - h)
        if int(horizon_mask.sum()) < 10:
            continue
        horizon_ends = ends[horizon_mask]
        actual = np.array([float(np.sum(tv[t : t + h])) for t in horizon_ends])

        forecasts_at_h: dict[str, np.ndarray] = {}
        for model, series in forecasts.items():
            uncond, persistence = aux[model]
            forecasts_at_h[model] = _cumulative_forecast(
                model, series[horizon_ends], uncond[horizon_ends], persistence[horizon_ends], h
            )

        for model, values in forecasts_at_h.items():
            valid = np.isfinite(actual) & np.isfinite(values)
            if int(valid.sum()) < 10:
                continue
            rows.append(
                {
                    "as_of": as_of, "run_id": run_id, "split": split, "model": model,
                    "metric": "qlike", "horizon": h,
                    "value": float(metrics.qlike(actual[valid], values[valid]).mean()),
                }
            )
            rows.append(
                {
                    "as_of": as_of, "run_id": run_id, "split": split, "model": model,
                    "metric": "rmse_vol", "horizon": h,
                    "value": float(
                        np.sqrt(np.mean((np.sqrt(actual[valid]) - np.sqrt(values[valid])) ** 2))
                    ),
                }
            )

        reference = forecasts_at_h.get("har")
        if reference is None:
            continue
        for model, values in forecasts_at_h.items():
            if model == "har":
                continue
            valid = np.isfinite(actual) & np.isfinite(values) & np.isfinite(reference)
            if int(valid.sum()) < 30:
                continue
            _, p_value = metrics.dm_test(
                metrics.qlike(actual[valid], values[valid]),
                metrics.qlike(actual[valid], reference[valid]),
                h=h,
            )
            rows.append(
                {
                    "as_of": as_of, "run_id": run_id, "split": split, "model": model,
                    "metric": "dm_p_vs_har", "horizon": h, "value": float(p_value),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run(
    settings: Settings,
    as_of: str | None = None,
    *,
    trigger: str = "manual",
    sink: Sink | None = None,
) -> dict:
    settings.validate()
    started = time.perf_counter()
    sink = sink or get_sink(settings)
    as_of = as_of or settings.as_of or dataset.resolve_as_of(settings)

    frame = sink.read_parquet(dataset.as_of_key(dataset.GOLD_FEATURES, as_of))
    params = sink.read_parquet(dataset.as_of_key(dataset.GOLD_PARAMS, as_of)).iloc[0]

    feature_names = [c for c in frame.columns if c not in (*TARGET_COLUMNS, "date")]
    data_fingerprint = str(params["data_fingerprint"])
    config_hash = settings.config_hash()
    run_id = make_run_id(as_of, config_hash, data_fingerprint)

    raw = frame[feature_names].to_numpy(dtype=float)
    returns = frame["ret"].to_numpy(dtype=float)
    park = frame["park"].to_numpy(dtype=float)

    c_train = float(params["c_train"])
    tau = float(params["tau"])
    tv = c_train * park
    log_tv = np.log(np.maximum(tv, EPS))
    label_high = (np.sqrt(np.maximum(tv, 0.0)) > tau).astype(float)

    i_tr = int(params["i_tr"])
    i_va = int(params["i_va"])
    n_real = int(params["n_real"])
    long_run = float(params["long_run_var_pct2"])
    persistence = float(params["persistence"])

    scaled, mean, scale = _standardise(raw[:i_tr], raw)

    X, y_var, y_regime, ends = seq.build_windows(scaled, log_tv, label_high, settings.window)

    train_mask = ends < i_tr
    val_mask = (ends >= i_tr) & (ends < i_va)
    test_mask = (ends >= i_va) & (ends < n_real)
    if train_mask.sum() < 100 or val_mask.sum() < 30:
        raise RuntimeError(
            f"data latih/validasi terlalu sedikit (train={int(train_mask.sum())}, "
            f"val={int(val_mask.sum())})"
        )

    config = seq.TrainConfig(
        hidden=settings.hidden,
        layers=settings.layers,
        dropout=settings.dropout,
        epochs=settings.epochs,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        patience=settings.patience,
        seed=settings.seed,
    )

    fitted = seq.fit(
        X[train_mask], y_var[train_mask], y_regime[train_mask],
        X[val_mask], y_var[val_mask], y_regime[val_mask],
        n_features=len(feature_names), config=config,
    )
    model = fitted["model"]

    # --- baseline walk-forward --------------------------------------------
    forecasts: dict[str, np.ndarray] = {
        "hist": baselines.historical_var(returns[:n_real]),
        "ewma": baselines.ewma_var(returns[:n_real], settings.ewma_lambda),
    }
    har = np.full(n_real, np.nan)
    har[i_tr:n_real] = baselines.har_log_walk_forward(
        tv[:n_real], i_tr, n_real, settings.refit_every
    )
    forecasts["har"] = har

    garch_uncond = np.full(n_real, np.nan)
    garch_persistence = np.full(n_real, np.nan)
    try:
        garch, uncond_series, persistence_series = baselines.garch_walk_forward(
            returns[:n_real], i_tr, n_real, "garch", settings.refit_every
        )
        garch_full = np.full(n_real, np.nan)
        garch_full[i_tr:n_real] = garch
        forecasts["garch"] = garch_full
        garch_uncond[i_tr:n_real] = uncond_series
        garch_persistence[i_tr:n_real] = persistence_series
    except Exception:  # noqa: BLE001 - arch tidak wajib untuk menjalankan pipeline
        pass

    # --- prediksi GRU walk-forward-in-sample -------------------------------
    gru_var = np.full(n_real, np.nan)
    gru_prob = np.full(n_real, np.nan)
    var_all, prob_all = seq.predict(model, X)
    gru_var[ends] = var_all
    gru_prob[ends] = prob_all
    forecasts["gru"] = gru_var

    # Ramalan h=1 untuk sesi berikutnya (baris fitur terakhir = hari depan).
    window_input = scaled[-settings.window :][None, :, :]
    gru_next_var, gru_next_prob = seq.predict(model, window_input)
    gru_next = float(gru_next_var[0])
    prob_next = float(gru_next_prob[0])

    # --- metrik h=1 --------------------------------------------------------
    metric_rows: list[dict] = []
    metric_rows += _metric_rows(ends[val_mask], tv, forecasts, as_of=as_of, run_id=run_id, split="val")
    metric_rows += _metric_rows(ends[test_mask], tv, forecasts, as_of=as_of, run_id=run_id, split="test")
    metric_rows += _dm_rows(ends[val_mask], tv, forecasts, as_of=as_of, run_id=run_id, split="val")
    metric_rows += _dm_rows(ends[test_mask], tv, forecasts, as_of=as_of, run_id=run_id, split="test")

    # --- metrik multi-horizon ---------------------------------------------
    aux: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "hist": (np.full(n_real, long_run), np.zeros(n_real)),
        "ewma": (np.full(n_real, long_run), np.zeros(n_real)),
        "har": (np.full(n_real, long_run), np.full(n_real, persistence)),
        "gru": (np.full(n_real, long_run), np.full(n_real, persistence)),
    }
    if "garch" in forecasts:
        aux["garch"] = (
            np.where(np.isfinite(garch_uncond), garch_uncond, long_run),
            np.where(np.isfinite(garch_persistence), garch_persistence, persistence),
        )
    for split_name, mask in (("val", val_mask), ("test", test_mask)):
        metric_rows += _multi_horizon_rows(
            ends[mask], tv, forecasts, aux,
            as_of=as_of, run_id=run_id, split=split_name,
            n_real=n_real, horizons=settings.horizons,
        )

    # --- model return + gate ----------------------------------------------
    ridge = Ridge(alpha=settings.ridge_alpha)
    train_ok = np.isfinite(returns[:i_tr]) & np.isfinite(raw[:i_tr]).all(axis=1)
    if int(train_ok.sum()) < 30:
        raise RuntimeError("baris latih untuk model return terlalu sedikit")
    ridge.fit(raw[:i_tr][train_ok], returns[:i_tr][train_ok])
    ret_val = ridge.predict(raw[i_tr:i_va])
    expanding_mean = pd.Series(returns[:n_real]).expanding(min_periods=30).mean().shift(1).to_numpy()
    val_r2 = metrics.r2_oos(returns[i_tr:i_va], ret_val, expanding_mean[i_tr:i_va])
    _, val_cw_p = metrics.clark_west(
        returns[i_tr:i_va], ret_val, expanding_mean[i_tr:i_va], expanding_mean[i_tr:i_va]
    )
    ml_return_enabled = bool(
        np.isfinite(val_r2) and val_r2 > 0
        and np.isfinite(val_cw_p) and val_cw_p < settings.ml_gate_alpha
    )

    ret_test = ridge.predict(raw[i_va:n_real])
    test_r2 = metrics.r2_oos(returns[i_va:n_real], ret_test, expanding_mean[i_va:n_real])
    _, test_cw_p = metrics.clark_west(
        returns[i_va:n_real], ret_test, expanding_mean[i_va:n_real], expanding_mean[i_va:n_real]
    )
    for split_name, value in (("val", val_r2), ("test", test_r2)):
        metric_rows.append(
            {"as_of": as_of, "run_id": run_id, "split": split_name, "model": "ridge",
             "metric": "r2_oos", "horizon": 1, "value": float(value)}
        )

    # --- kandidat + pemilihan model ---------------------------------------
    val_qlike = {
        row["model"]: row["value"]
        for row in metric_rows
        if row["split"] == "val" and row["metric"] == "qlike" and row["horizon"] == 1
    }
    test_qlike = {
        row["model"]: row["value"]
        for row in metric_rows
        if row["split"] == "test" and row["metric"] == "qlike" and row["horizon"] == 1
    }

    har_next = baselines.har_forecast_next(tv, n_real)
    hist_next = baselines.hist_forecast_next(returns, n_real)
    ewma_next = baselines.ewma_forecast_next(returns, n_real, settings.ewma_lambda)

    candidates_rows: list[dict] = [
        {
            "as_of": as_of, "run_id": run_id, "model": "gru",
            "val_qlike": val_qlike.get("gru", np.nan),
            "test_qlike": test_qlike.get("gru", np.nan),
            "h1_var": gru_next, "uncond": long_run, "persistence": persistence,
            "prob_high": prob_next, "selected": False,
        },
        {
            "as_of": as_of, "run_id": run_id, "model": "har",
            "val_qlike": val_qlike.get("har", np.nan),
            "test_qlike": test_qlike.get("har", np.nan),
            "h1_var": har_next, "uncond": long_run, "persistence": persistence,
            "prob_high": np.nan, "selected": False,
        },
        {
            "as_of": as_of, "run_id": run_id, "model": "ewma",
            "val_qlike": val_qlike.get("ewma", np.nan),
            "test_qlike": test_qlike.get("ewma", np.nan),
            "h1_var": ewma_next, "uncond": long_run, "persistence": 0.0,
            "prob_high": np.nan, "selected": False,
        },
        {
            "as_of": as_of, "run_id": run_id, "model": "hist",
            "val_qlike": val_qlike.get("hist", np.nan),
            "test_qlike": test_qlike.get("hist", np.nan),
            "h1_var": hist_next, "uncond": long_run, "persistence": 0.0,
            "prob_high": np.nan, "selected": False,
        },
    ]
    if "garch" in forecasts:
        try:
            garch_next, garch_next_uncond, garch_next_persistence = baselines.garch_forecast_next(
                returns, n_real, "garch"
            )
            candidates_rows.append(
                {
                    "as_of": as_of, "run_id": run_id, "model": "garch",
                    "val_qlike": val_qlike.get("garch", np.nan),
                    "test_qlike": test_qlike.get("garch", np.nan),
                    "h1_var": garch_next, "uncond": garch_next_uncond,
                    "persistence": garch_next_persistence,
                    "prob_high": np.nan, "selected": False,
                }
            )
        except Exception:  # noqa: BLE001
            pass

    candidates = pd.DataFrame(candidates_rows)
    comparable = candidates[candidates["val_qlike"].notna()]
    if comparable.empty:
        raise RuntimeError("tidak ada kandidat dengan QLIKE validasi yang valid")
    selected_model = str(comparable.loc[comparable["val_qlike"].idxmin(), "model"])
    selected_qlike = float(comparable["val_qlike"].min())
    candidates["selected"] = candidates["model"] == selected_model
    validate_contract(candidates, GOLD_CANDIDATES)
    sink.write_parquet(candidates, dataset.as_of_key(dataset.GOLD_CANDIDATES, as_of), overwrite=True)

    # --- keputusan promosi -------------------------------------------------
    champion_frame = sink.try_read_parquet(dataset.GOLD_CHAMPION)
    previous_qlike = (
        float(champion_frame["val_qlike"].iloc[0])
        if champion_frame is not None and not champion_frame.empty
        else float("inf")
    )
    previous_run_id = (
        str(champion_frame["run_id"].iloc[0])
        if champion_frame is not None and not champion_frame.empty
        else None
    )
    # Promosi hanya bila modelnya BERBEDA dan tidak lebih buruk. run_id
    # deterministik, jadi melatih ulang as_of+config yang sama adalah no-op:
    # tidak menambah versi registry, tidak mengubah champion.
    promoted = bool(
        np.isfinite(selected_qlike)
        and selected_qlike <= previous_qlike
        and previous_run_id != run_id
    )

    # --- evaluasi untuk dashboard -----------------------------------------
    test_ends = ends[test_mask]
    regime_eval = pd.DataFrame({
        "as_of": as_of,
        "run_id": run_id,
        "date": frame["date"].to_numpy()[test_ends],
        "prob_high": gru_prob[test_ends],
        "predicted_high": gru_prob[test_ends] >= 0.5,
        "actual_high": label_high[test_ends] > 0.5,
    })
    regime_eval["correct"] = regime_eval["predicted_high"] == regime_eval["actual_high"]

    vol_eval = pd.DataFrame({
        "as_of": as_of,
        "run_id": run_id,
        "date": frame["date"].to_numpy()[test_ends],
        "actual_var": tv[test_ends],
        "gru_var": gru_var[test_ends],
        "har_var": har[test_ends],
        "ewma_var": forecasts["ewma"][test_ends],
        "hist_var": forecasts["hist"][test_ends],
    })

    test_selected_qlike = float(test_qlike.get(selected_model, float("nan")))
    regime = metrics.regime_metrics(regime_eval["actual_high"], regime_eval["prob_high"])

    # --- tulis artefak -----------------------------------------------------
    import pyarrow as pa
    import pyarrow.parquet as pq

    model_blob = seq.dumps(model, config, settings.window, len(feature_names), feature_names)
    scaler_frame = pd.DataFrame({"feature": feature_names, "mean": mean, "scale": scale})
    scaler_buffer = io.BytesIO()
    pq.write_table(
        pa.Table.from_pandas(scaler_frame.reset_index(drop=True), preserve_index=False),
        scaler_buffer,
    )
    ridge_buffer = io.BytesIO()
    import joblib

    joblib.dump({"model": ridge, "features": feature_names}, ridge_buffer)

    artifacts = {
        "model.pt": model_blob,
        "scaler.parquet": scaler_buffer.getvalue(),
        "ridge.joblib": ridge_buffer.getvalue(),
    }
    for name, blob in artifacts.items():
        sink.write_bytes(blob, dataset.artifact_key(run_id, name), overwrite=True)

    metrics_frame = pd.DataFrame(metric_rows)
    validate_contract(metrics_frame, GOLD_METRICS)
    sink.write_parquet(metrics_frame, dataset.as_of_key(dataset.GOLD_METRICS, as_of), overwrite=True)

    validate_contract(regime_eval, GOLD_REGIME_EVAL)
    sink.write_parquet(regime_eval, dataset.as_of_key(dataset.GOLD_REGIME_EVAL, as_of), overwrite=True)

    validate_contract(vol_eval, GOLD_VOL_EVAL)
    sink.write_parquet(vol_eval, dataset.as_of_key(dataset.GOLD_VOL_EVAL, as_of), overwrite=True)

    duration = time.perf_counter() - started
    append_run(
        sink,
        as_of=as_of,
        run_id=run_id,
        stage="train",
        trigger=trigger,
        status="ok",
        promoted=promoted,
        val_metric=selected_qlike,
        test_metric=test_selected_qlike,
        duration_s=duration,
        data_fingerprint=data_fingerprint,
        config_hash=config_hash,
    )


    # --- pencatatan MLflow (opsional, tidak boleh menggagalkan training) ----
    selected_row = candidates[candidates["model"] == selected_model].iloc[0]
    ml_metrics: dict[str, float | None] = {}
    for row in candidates_rows:
        ml_metrics[f"val_qlike_{row['model']}"] = row["val_qlike"]
        ml_metrics[f"test_qlike_{row['model']}"] = row["test_qlike"]
    for row in metric_rows:
        if row["split"] == "test" and row["horizon"] > 1 and row["metric"] == "qlike":
            ml_metrics[f"test_qlike_{row['model']}_h{row['horizon']}"] = row["value"]
    ml_metrics.update({
        "regime_auc": regime["auc"],
        "regime_precision": regime["precision"],
        "regime_recall": regime["recall"],
        "regime_f1": regime["f1"],
        "regime_base_rate": regime["base_rate"],
        "val_r2_oos": val_r2,
        "test_r2_oos": test_r2,
        "val_clark_west_p": val_cw_p,
        "test_clark_west_p": test_cw_p,
    })

    tracking_info = tracking.log_training_run(
        as_of=as_of,
        run_id=run_id,
        trigger=trigger,
        selected_model=selected_model,
        promoted=promoted,
        params={
            **settings.model_config(),
            "n_features": len(feature_names),
            "selected_model": selected_model,
            "ml_return_enabled": ml_return_enabled,
            "train_days": i_tr,
            "val_days": i_va - i_tr,
            "test_days": n_real - i_va,
            "tau": tau,
            "c_train": c_train,
        },
        metrics=ml_metrics,
        tags={
            "ihsg_run_id": run_id,
            "as_of": as_of,
            "data_fingerprint": data_fingerprint,
            "config_hash": config_hash,
            "git_sha": git_sha(),
        },
        artifacts=artifacts,
        recipe={
            "model": selected_model,
            "as_of": as_of,
            "run_id": run_id,
            "horizons": int(settings.horizons),
            "h1_var": float(selected_row["h1_var"]),
            "uncond": float(selected_row["uncond"]),
            "persistence": float(selected_row["persistence"]),
            "tau": tau,
            "prob_high": prob_next,
        } if promoted else None,
    )

    # --- champion (ditulis setelah tracking agar versi registry ikut tercatat)
    if promoted:
        champion = pd.DataFrame([{
            "run_id": run_id,
            "trained_as_of": as_of,
            "model_vol": selected_model,
            "model_ret": "ridge" if ml_return_enabled else "expanding_mean",
            "val_qlike": selected_qlike,
            "val_r2_oos": float(val_r2),
            "ml_return_enabled": ml_return_enabled,
            "tau": tau,
            "c_train": c_train,
            "long_run_var_pct2": long_run,
            "persistence": persistence,
            "window": int(settings.window),
            "horizons": int(settings.horizons),
            "n_features": int(len(feature_names)),
            "config_hash": config_hash,
            "data_fingerprint": data_fingerprint,
            "mlflow_run_id": str(tracking_info.get("mlflow_run_id") or ""),
            "mlflow_version": str(tracking_info.get("model_version") or ""),
        }])
        validate_contract(champion, GOLD_CHAMPION)
        sink.write_parquet(champion, dataset.GOLD_CHAMPION, overwrite=True)

    return {
        "stage": "train",
        "as_of": as_of,
        "run_id": run_id,
        "status": "ok",
        "trigger": trigger,
        "promoted": promoted,
        "selected_model": selected_model,
        "candidates": {
            row["model"]: round(float(row["val_qlike"]), 6)
            for row in candidates_rows
            if np.isfinite(row["val_qlike"])
        },
        "val_qlike": round(float(selected_qlike), 6),
        "test_qlike": None if not np.isfinite(test_selected_qlike) else round(test_selected_qlike, 6),
        "ml_return_enabled": ml_return_enabled,
        "val_r2_oos": round(float(val_r2), 6),
        "test_r2_oos": round(float(test_r2), 6),
        "val_clark_west_p": None if not np.isfinite(val_cw_p) else round(float(val_cw_p), 6),
        "test_clark_west_p": None if not np.isfinite(test_cw_p) else round(float(test_cw_p), 6),
        "previous_champion_qlike": previous_qlike if np.isfinite(previous_qlike) else None,
        "best_epoch": fitted["best_epoch"],
        "regime": {k: round(v, 4) for k, v in regime.items() if np.isfinite(v)},
        "duration_s": round(duration, 1),
        "finished_at": datetime.now(ZoneInfo(JAKARTA_TZ)).isoformat(timespec="seconds"),
        "features_fingerprint": fingerprint(frame),
        "tracking": tracking_info,
    }
