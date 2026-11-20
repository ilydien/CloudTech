"""Flow harian: ingest -> silver -> features -> predict.

Tidak ada training di sini. Model yang dipakai adalah champion hasil flow
mingguan (atau backfill), sehingga job harian selesai dalam hitungan detik.
"""

from __future__ import annotations

from prefect import flow, get_run_logger

from ihsg.config import Settings
from ihsg.pipelines import features as features_pipeline
from ihsg.pipelines import ingest as ingest_pipeline
from ihsg.pipelines import predict as predict_pipeline
from ihsg.pipelines import silver as silver_pipeline


@flow(name="ihsg-daily-predict", log_prints=True)
def daily_pipeline(as_of: str | None = None) -> dict:
    logger = get_run_logger()
    settings = Settings.from_env()
    settings.validate()

    ingest_result = ingest_pipeline.run(settings, as_of)
    logger.info("ingest: %s (%s)", ingest_result["status"], ingest_result["as_of"])

    silver_result = silver_pipeline.run(settings, ingest_result["as_of"])
    logger.info("silver: %s baris, %s DQ gagal", silver_result["rows"], silver_result["dq_failed"])

    features_result = features_pipeline.run(settings, ingest_result["as_of"])
    logger.info("features: %s baris, %s fitur", features_result["rows"], features_result["features"])

    predict_result = predict_pipeline.run(settings, ingest_result["as_of"])
    logger.info(
        "predict: vol H+1 %.3f%%, rezim %s, median 7 hari %.2f",
        predict_result["volatility_h1_pct"],
        predict_result["regime_h1"],
        predict_result["median_close_7d"],
    )

    return {
        "as_of": ingest_result["as_of"],
        "ingest": ingest_result,
        "silver": silver_result,
        "features": features_result,
        "predict": predict_result,
    }


if __name__ == "__main__":
    daily_pipeline()
