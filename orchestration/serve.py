"""Penjadwalan: dua deployment, satu proses `serve`.

Dijalankan sebagai container `prefect-serve`. Proses ini mendaftarkan deployment
ke server Prefect (sehingga jadwal & riwayat terlihat di UI) **sekaligus**
mengeksekusi flow saat jadwal tiba. Tidak perlu work pool dan tidak perlu
`prefect deploy`, karena kode flow sudah ada di dalam image.

Jadwal (WIB):
  ihsg-daily-predict   05:15 Senin-Jumat
  ihsg-weekly-retrain  06:00 Sabtu

Flow-nya tipis dan semuanya idempoten, jadi menjalankan ulang jadwal yang sama
tidak menggandakan data (lihat `ihsg/runlog.py` dan `ihsg/pipelines/*`).
"""

from __future__ import annotations

from prefect import serve
from prefect.client.schemas.schedules import CronSchedule

from flows.daily_predict import daily_pipeline
from flows.weekly_retrain import weekly_pipeline

JAKARTA = "Asia/Jakarta"

if __name__ == "__main__":
    serve(
        daily_pipeline.to_deployment(
            name="ihsg-daily-predict",
            description=(
                "Ingest incremental, silver, fitur, latih bila kandidat hari itu "
                "belum ada, lalu ramalan 1-7 hari. Berjalan setelah pasar AS tutup "
                "dan sebelum BEI buka."
            ),
            schedules=[CronSchedule(cron="15 5 * * 1-5", timezone=JAKARTA)],
            concurrency_limit=1,
            tags=["ihsg", "daily"],
        ),
        weekly_pipeline.to_deployment(
            name="ihsg-weekly-retrain",
            description=(
                "Latih ulang semua kandidat, evaluasi di validasi, promosikan hanya "
                "bila QLIKE membaik, lalu perbarui ramalan."
            ),
            schedules=[CronSchedule(cron="0 6 * * 6", timezone=JAKARTA)],
            concurrency_limit=1,
            tags=["ihsg", "retrain"],
        ),
    )
