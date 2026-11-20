"""Gold/train: latih GRU, evaluasi terhadap baseline, dan putuskan promosi.

Pola **champion-challenger**:
* model baru (challenger) dievaluasi di data validasi,
* ia hanya menggantikan champion bila QLIKE validasi lebih baik,
* keputusan promosi tetap dicatat walau kalah - inilah yang membuat riwayat
  retrain bermakna (dan memenuhi bukti "retrain lebih dari 3x").

``run_id`` deterministik: ``sha256(as_of | config_hash | data_fingerprint)``.
Retrain untuk tanggal & konfigurasi yang sama menghasilkan run_id yang sama,
sehingga penulisan ulang bersifat idempoten dan bukan duplikat palsu.
"""

from __future__ import annotations

import hashlib
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
from ihsg.schemas import (
    GOLD_CHAMPION,
    GOLD_METRICS,
    GOLD_REGIME_EVAL,
    GOLD_RUN_LOG,
    GOLD_VOL_EVAL,
    fingerprint,
    validate_contract,
)
from ihsg.storage import Sink, get_sink

EPS = 1e-6


def make_run_id(as_of: str, config_hash: str, data_fingerprint: str) -> str:
    payload = f"{as_of}|{config_hash}|{data_fingerprint}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _git_sha() -> str:
    """Stempel versi kode; kosong bila .git tidak tersedia di image."""
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _standardise(train: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0, ddof=0)
    scale[scale < 1e-9] = 1.0
    return (data - mean) / scale, mean, scale


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
    """p-value Diebold-Mariano challenger vs tiap baseline (loss = QLIKE).

    p kecil + statistik negatif = challenger lebih baik. Disimpan sebagai
    ``metric='dm_p_vs_<bench>'`` agar tetap satu tabel dengan metrik lain.
    """
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
    run_id = make_run_id(as_of, settings.config_hash(), data_fingerprint)

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

    # --- baseline pada skala & periode yang sama ---------------------------
    forecasts: dict[str, np.ndarray] = {
        "hist": baselines.historical_var(returns[:n_real]),
        "ewma": baselines.ewma_var(returns[:n_real], settings.ewma_lambda),
    }
    har = np.full(n_real, np.nan)
    har[i_tr:n_real] = baselines.har_log_walk_forward(
        tv[:n_real], i_tr, n_real, settings.refit_every
    )
    forecasts["har"] = har

    try:
        garch, _, _ = baselines.garch_walk_forward(
            returns[:n_real], i_tr, n_real, "garch", settings.refit_every
        )
        garch_full = np.full(n_real, np.nan)
        garch_full[i_tr:n_real] = garch
        forecasts["garch"] = garch_full
    except Exception:  # noqa: BLE001 - arch tidak wajib untuk menjalankan pipeline
        pass

    # --- prediksi model ----------------------------------------------------
    gru_var = np.full(n_real, np.nan)
    gru_prob = np.full(n_real, np.nan)
    var_all, prob_all = seq.predict(model, X)
    gru_var[ends] = var_all
    gru_prob[ends] = prob_all
    forecasts["gru"] = gru_var

    metric_rows: list[dict] = []
    metric_rows += _metric_rows(ends[val_mask], tv, forecasts, as_of=as_of, run_id=run_id, split="val")
    metric_rows += _metric_rows(ends[test_mask], tv, forecasts, as_of=as_of, run_id=run_id, split="test")
    metric_rows += _dm_rows(ends[val_mask], tv, forecasts, as_of=as_of, run_id=run_id, split="val")
    metric_rows += _dm_rows(ends[test_mask], tv, forecasts, as_of=as_of, run_id=run_id, split="test")

    val_qlike = next(
        (row["value"] for row in metric_rows
         if row["split"] == "val" and row["model"] == "gru" and row["metric"] == "qlike"),
        float("nan"),
    )

    # --- model return (gate produksi) --------------------------------------
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
    ml_return_enabled = bool(np.isfinite(val_r2) and val_r2 > 0 and np.isfinite(val_cw_p) and val_cw_p < settings.ml_gate_alpha)

    # Metrik model return (dilaporkan terpisah dari metrik volatilitas).
    ret_test = ridge.predict(raw[i_va:n_real])
    test_r2 = metrics.r2_oos(returns[i_va:n_real], ret_test, expanding_mean[i_va:n_real])
    _, test_cw_p = metrics.clark_west(
        returns[i_va:n_real], ret_test, expanding_mean[i_va:n_real], expanding_mean[i_va:n_real]
    )
    metric_rows.append(
        {"as_of": as_of, "run_id": run_id, "split": "val", "model": "ridge",
         "metric": "r2_oos", "horizon": 1, "value": float(val_r2)}
    )
    metric_rows.append(
        {"as_of": as_of, "run_id": run_id, "split": "test", "model": "ridge",
         "metric": "r2_oos", "horizon": 1, "value": float(test_r2)}
    )

    # --- keputusan promosi -------------------------------------------------
    champion_frame = sink.try_read_parquet(dataset.GOLD_CHAMPION)
    previous_qlike = (
        float(champion_frame["val_qlike"].iloc[0])
        if champion_frame is not None and not champion_frame.empty
        else float("inf")
    )
    promoted = bool(np.isfinite(val_qlike) and val_qlike <= previous_qlike)

    # --- evaluasi yang dipakai dashboard -----------------------------------
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

    test_qlike = next(
        (row["value"] for row in metric_rows
         if row["split"] == "test" and row["model"] == "gru" and row["metric"] == "qlike"),
        float("nan"),
    )
    regime = metrics.regime_metrics(regime_eval["actual_high"], regime_eval["prob_high"])

    # --- tulis artefak -----------------------------------------------------
    sink.write_bytes(
        seq.dumps(model, config, settings.window, len(feature_names), feature_names),
        dataset.artifact_key(run_id, "model.pt"),
        overwrite=True,
    )
    sink.write_parquet(
        pd.DataFrame({"feature": feature_names, "mean": mean, "scale": scale}),
        dataset.artifact_key(run_id, "scaler.parquet"),
        overwrite=True,
    )

    import io

    import joblib

    buffer = io.BytesIO()
    joblib.dump({"model": ridge, "features": feature_names}, buffer)
    sink.write_bytes(buffer.getvalue(), dataset.artifact_key(run_id, "ridge.joblib"), overwrite=True)

    metrics_frame = pd.DataFrame(metric_rows)
    validate_contract(metrics_frame, GOLD_METRICS)
    sink.write_parquet(metrics_frame, dataset.as_of_key(dataset.GOLD_METRICS, as_of), overwrite=True)

    validate_contract(regime_eval, GOLD_REGIME_EVAL)
    sink.write_parquet(regime_eval, dataset.as_of_key(dataset.GOLD_REGIME_EVAL, as_of), overwrite=True)

    validate_contract(vol_eval, GOLD_VOL_EVAL)
    sink.write_parquet(vol_eval, dataset.as_of_key(dataset.GOLD_VOL_EVAL, as_of), overwrite=True)

    duration = time.perf_counter() - started
    run_log = pd.DataFrame([{
        "as_of": as_of,
        "run_id": run_id,
        "trigger": trigger,
        "stage": "train",
        "status": "ok",
        "promoted": promoted,
        "val_metric": float(val_qlike),
        "test_metric": float(test_qlike),
        "duration_s": float(duration),
        "data_fingerprint": data_fingerprint,
        "config_hash": settings.config_hash(),
        "git_sha": _git_sha(),
    }])
    validate_contract(run_log, GOLD_RUN_LOG)
    existing = sink.try_read_parquet(dataset.as_of_key(dataset.GOLD_RUN_LOG, as_of))
    combined = run_log if existing is None else pd.concat([existing, run_log], ignore_index=True)
    combined = combined.drop_duplicates(subset=["as_of", "run_id"], keep="last")
    sink.write_parquet(combined, dataset.as_of_key(dataset.GOLD_RUN_LOG, as_of), overwrite=True)

    if promoted:
        champion = pd.DataFrame([{
            "run_id": run_id,
            "trained_as_of": as_of,
            "model_vol": "gru",
            "model_ret": "ridge" if ml_return_enabled else "expanding_mean",
            "val_qlike": float(val_qlike),
            "val_r2_oos": float(val_r2),
            "ml_return_enabled": ml_return_enabled,
            "tau": tau,
            "c_train": c_train,
            "long_run_var_pct2": float(params["long_run_var_pct2"]),
            "persistence": float(params["persistence"]),
            "window": int(settings.window),
            "horizons": int(settings.horizons),
            "n_features": int(len(feature_names)),
            "config_hash": settings.config_hash(),
            "data_fingerprint": data_fingerprint,
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
        "ml_return_enabled": ml_return_enabled,
        "val_qlike": round(float(val_qlike), 6),
        "test_qlike": round(float(test_qlike), 6),
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
    }
