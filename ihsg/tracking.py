"""Integrasi MLflow: pencatatan run + registry champion.

Prinsip yang dijaga:

* **MLflow bersifat deskriptif.** Sumber kebenaran operasional tetap
  ``gold/champion`` + ``gold/candidates`` (Parquet) yang dibaca oleh serving.
  MLflow dipakai untuk eksplorasi, perbandingan run, dan versioning model.
  Dengan begitu, MLflow mati **tidak** menghentikan produksi.
* **Kegagalan pencatatan tidak menggagalkan training.** Semua panggilan dibungkus
  try/except; pesan peringatan ditulis ke stderr, run tetap selesai.
* **Artefak tidak diduplikasi tanpa alasan.** File yang dikirim ke MLflow adalah
  tiga artefak kecil (model.pt, scaler.parquet, ridge.joblib) yang sudah ada di
  ``s3://mlartifacts/run_id=...``. MLflow menyimpannya di
  ``s3://mlartifacts/mlflow`` agar semuanya tetap di Garage.

Aktif hanya bila ``MLFLOW_TRACKING_URI`` di-set.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

EXPERIMENT_NAME = "ihsg-volatility"
REGISTERED_MODEL_NAME = "ihsg-champion"
CHAMPION_ALIAS = "champion"


def tracking_uri() -> str | None:
    value = os.getenv("MLFLOW_TRACKING_URI")
    if value is None or not value.strip():
        return None
    return value.strip()


def enabled() -> bool:
    return tracking_uri() is not None


def _as_param(value: object) -> str | int | float:
    """MLflow hanya menerima str/int/float sebagai param."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value
    return str(value)


def _finite(metrics: dict[str, float | None]) -> dict[str, float]:
    clean: dict[str, float] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        number = float(value)
        if np.isfinite(number):
            clean[key] = number
    return clean


def log_training_run(
    *,
    as_of: str,
    run_id: str,
    trigger: str,
    selected_model: str,
    promoted: bool,
    params: dict[str, object],
    metrics: dict[str, float | None],
    tags: dict[str, str],
    artifacts: dict[str, bytes],
    recipe: dict | None = None,
) -> dict:
    """Catat satu run ke MLflow. Return dict ringkasan (kosong bila nonaktif)."""
    uri = tracking_uri()
    if uri is None:
        return {"enabled": False, "reason": "MLFLOW_TRACKING_URI tidak di-set"}

    try:
        import mlflow
        from mlflow import MlflowClient

        from ihsg.champion_model import ChampionRecipe

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name=f"{as_of}-{selected_model}") as run:
            mlflow.set_tags(
                {
                    **tags,
                    "ihsg_run_id": run_id,
                    "ihsg_as_of": as_of,
                    "ihsg_trigger": trigger,
                    "ihsg_selected_model": selected_model,
                    "ihsg_promoted": str(bool(promoted)),
                }
            )
            mlflow.log_params({key: _as_param(value) for key, value in params.items()})

            clean_metrics = _finite(metrics)
            if clean_metrics:
                mlflow.log_metrics(clean_metrics)

            if artifacts:
                with tempfile.TemporaryDirectory() as directory:
                    for name, blob in artifacts.items():
                        (Path(directory) / name).write_bytes(blob)
                    mlflow.log_artifacts(directory, artifact_path="artifacts")

            model_version: int | None = None
            if promoted and recipe is not None:
                mlflow.pyfunc.log_model(
                    artifact_path="champion",
                    python_model=ChampionRecipe(recipe),
                    code_paths=[str(Path(__file__).resolve().parent)],
                )
                registered = mlflow.register_model(
                    model_uri=f"runs:/{run.info.run_id}/champion",
                    name=REGISTERED_MODEL_NAME,
                )
                model_version = int(registered.version)
                MlflowClient().set_registered_model_alias(
                    REGISTERED_MODEL_NAME, CHAMPION_ALIAS, model_version
                )

            return {
                "enabled": True,
                "experiment": EXPERIMENT_NAME,
                "mlflow_run_id": run.info.run_id,
                "model_name": REGISTERED_MODEL_NAME if model_version else None,
                "model_version": model_version,
                "alias": CHAMPION_ALIAS if model_version else None,
                "tracking_uri": uri,
            }

    except Exception as error:  # noqa: BLE001 - pencatatan tidak boleh menggagalkan training
        print(
            f"[mlflow] peringatan: pencatatan run gagal ({type(error).__name__}: {error})",
            file=sys.stderr,
        )
        return {"enabled": False, "error": f"{type(error).__name__}: {error}"}
