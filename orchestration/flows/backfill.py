"""Flow backfill manual: jalankan pipeline untuk rentang tanggal.

Dipakai untuk:
* mengisi riwayat prediksi (mis. mereproduksi evaluasi),
* menghasilkan beberapa run retrain sebagai bukti (bonus "retrain > 3x").
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


@flow(name="ihsg-backfill", log_prints=True)
def backfill_pipeline(
    start: str | None = None,
    end: str | None = None,
    retrain: bool = True,
) -> dict:
    logger = get_run_logger()
    settings = Settings.from_env()
    settings.validate()

    start = start or settings.start_date
    end = end or dataset.resolve_as_of(settings)
    days = [day.isoformat() for day in dataset.trading_days(start, end, settings.idx_holidays).date]

    results: list[dict] = []
    for as_of in days:
        record: dict = {"requested_as_of": as_of}
        try:
            ingest_result = ingest_pipeline.run(settings, as_of)
            resolved = ingest_result["as_of"]
            record["as_of"] = resolved
            silver_pipeline.run(settings, resolved)
            features_pipeline.run(settings, resolved)
            if retrain:
                record["train"] = train_pipeline.run(settings, resolved, trigger="backfill")
            record["predict"] = predict_pipeline.run(settings, resolved, trigger="backfill")
            record["status"] = "ok"
        except Exception as error:  # noqa: BLE001 - backfill harus lanjut
            record["status"] = "failed"
            record["error"] = f"{type(error).__name__}: {error}"
            logger.warning("backfill %s gagal: %s", as_of, record["error"])
        results.append(record)

    ok = sum(1 for item in results if item["status"] == "ok")
    logger.info("backfill selesai: %s/%s hari", ok, len(results))
    return {"days": len(results), "ok": ok, "failed": len(results) - ok, "results": results}


if __name__ == "__main__":
    backfill_pipeline()
