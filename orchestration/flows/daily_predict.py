"""Flow harian: ingest -> silver -> features -> predict.

Tidak ada training di sini. Model yang dipakai adalah champion hasil flow
mingguan (atau backfill), sehingga job harian selesai dalam hitungan detik.
"""

from __future__ import annotations

from prefect import flow, get_run_logger

from ihsg import dataset
from ihsg.config import Settings
from ihsg.pipelines import features as features_pipeline
from ihsg.pipelines import ingest as ingest_pipeline
from ihsg.pipelines import predict as predict_pipeline
from ihsg.pipelines import silver as silver_pipeline
from ihsg.pipelines import train as train_pipeline
from ihsg.storage import get_sink


@flow(name="ihsg-daily-predict", log_prints=True)
def daily_pipeline(as_of: str | None = None) -> dict:
    logger = get_run_logger()
    settings = Settings.from_env()
    settings.validate()
    sink = get_sink(settings)

    ingest_result = ingest_pipeline.run(settings, as_of)
    resolved = ingest_result["as_of"]
    logger.info("ingest: %s (%s)", ingest_result["status"], resolved)

    silver_result = silver_pipeline.run(settings, resolved)
    logger.info("silver: %s baris, %s DQ gagal", silver_result["rows"], silver_result["dq_failed"])

    features_result = features_pipeline.run(settings, resolved)
    logger.info("features: %s baris, %s fitur", features_result["rows"], features_result["features"])

    # predict butuh gold/candidates; latih bila kandidat hari itu belum ada.
    needs_training = not sink.exists(dataset.as_of_key(dataset.GOLD_CANDIDATES, resolved))
    train_result = None
    if needs_training:
        train_result = train_pipeline.run(settings, resolved, trigger="schedule")
        logger.info(
            "train: model=%s val_qlike=%.4f test_qlike=%.4f promoted=%s",
            train_result["selected_model"],
            train_result["val_qlike"],
            train_result["test_qlike"] or float("nan"),
            train_result["promoted"],
        )

    predict_result = predict_pipeline.run(settings, resolved, trigger="schedule")
    logger.info(
        "predict: model=%s vol H+1 %.3f%%, rezim %s, median 7 hari %.2f",
        predict_result["model_vol"],
        predict_result["volatility_h1_pct"],
        predict_result["regime_h1"],
        predict_result["median_close_7d"],
    )

    return {
        "as_of": resolved,
        "trained": needs_training,
        "ingest": ingest_result,
        "silver": silver_result,
        "features": features_result,
        "train": train_result,
        "predict": predict_result,
    }


if __name__ == "__main__":
    daily_pipeline()
