"""Riwayat run per tahap -> ``gold/run_log``.

Satu tabel per ``as_of`` (append + dedupe). Ini sumber bukti bahwa pipeline
benar-benar berjalan otomatis: setiap tahap meninggalkan jejak, bukan hanya
training.

Kenapa bukan UI orkestrator: bukti harus berupa **data** (Parquet) yang bisa
dilampirkan di laporan, bukan screenshot yang bergantung pada server Prefect.
"""

from __future__ import annotations

import subprocess

import pandas as pd

from ihsg import dataset
from ihsg.schemas import GOLD_RUN_LOG, validate_contract
from ihsg.storage import Sink


def git_sha() -> str:
    """Stempel versi kode; 'unknown' bila .git tidak ada di image."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def append_run(
    sink: Sink,
    *,
    as_of: str,
    run_id: str,
    stage: str,
    trigger: str,
    status: str = "ok",
    promoted: bool = False,
    val_metric: float | None = None,
    test_metric: float | None = None,
    duration_s: float = 0.0,
    data_fingerprint: str = "",
    config_hash: str = "",
) -> str:
    """Tulis satu baris riwayat. Return key yang ditulis."""
    row = pd.DataFrame(
        [
            {
                "as_of": as_of,
                "run_id": run_id,
                "trigger": trigger,
                "stage": stage,
                "status": status,
                "promoted": bool(promoted),
                "val_metric": float("nan") if val_metric is None else float(val_metric),
                "test_metric": float("nan") if test_metric is None else float(test_metric),
                "duration_s": float(duration_s),
                "data_fingerprint": data_fingerprint,
                "config_hash": config_hash,
                "git_sha": git_sha(),
            }
        ]
    )
    validate_contract(row, GOLD_RUN_LOG)

    key = dataset.as_of_key(dataset.GOLD_RUN_LOG, as_of)
    existing = sink.try_read_parquet(key)
    combined = row if existing is None else pd.concat([existing, row], ignore_index=True)
    combined = combined.drop_duplicates(subset=["as_of", "stage", "run_id"], keep="last")
    sink.write_parquet(combined, key, overwrite=True)
    return key
