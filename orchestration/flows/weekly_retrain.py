"""Flow retrain mingguan: data segar -> latih ulang -> gate -> promosi -> predict.

Pola champion-challenger: ``train`` hanya menimpa ``gold/champion`` bila QLIKE
validasi challenger lebih baik. Kalau kalah, hasilnya tetap tercatat di
``gold/run_log`` - inilah bukti retrain berulang yang bisa diaudit.
"""

from __future__ import annotations

from prefect import flow, get_run_logger

from ihsg.config import Settings
from ihsg.pipelines import features as features_pipeline
from ihsg.pipelines import ingest as ingest_pipeline
from ihsg.pipelines import predict as predict_pipeline
from ihsg.pipelines import silver as silver_pipeline
from ihsg.pipelines import train as train_pipeline


@flow(name="ihsg-weekly-retrain", log_prints=True)
def weekly_pipeline(as_of: str | None = None) -> dict:
    logger = get_run_logger()
    settings = Settings.from_env()
    settings.validate()

    ingest_result = ingest_pipeline.run(settings, as_of)
    resolved = ingest_result["as_of"]
    silver_pipeline.run(settings, resolved)
    features_pipeline.run(settings, resolved)

    train_result = train_pipeline.run(settings, resolved, trigger="schedule")
    logger.info(
        "train: run_id=%s promoted=%s val_qlike=%.4f test_qlike=%.4f",
        train_result["run_id"],
        train_result["promoted"],
        train_result["val_qlike"],
        train_result["test_qlike"],
    )

    predict_result = predict_pipeline.run(settings, resolved)
    return {"train": train_result, "predict": predict_result}


if __name__ == "__main__":
    weekly_pipeline()
