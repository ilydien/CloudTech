"""Riwayat run per tahap -> ``gold/run_log``.

Desain: **satu baris = satu file**, dengan kunci
``gold/run_log/as_of=<D>/stage=<S>/run_id=<R>/data.parquet``.

Kenapa tidak satu file per ``as_of`` (baca-ubah-tulis): pola itu rawan
*lost update* bila dua tahap menulis bersamaan (baca daftar lama, gabung, tulis).
Dengan satu file per entri, penulisan menjadi **upsert murni**: menulis ulang
kunci yang sama = idempoten, dan tidak pernah menyentuh baris lain.
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


def run_key(as_of: str, stage: str, run_id: str) -> str:
    return (
        f"{dataset.GOLD_RUN_LOG}/as_of={as_of}/stage={stage}"
        f"/run_id={run_id}/{dataset.PART_FILE}"
    )


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

    key = run_key(as_of, stage, run_id)
    sink.write_parquet(row, key, overwrite=True)
    return key


def read_runs(sink: Sink, as_of: str | None = None) -> pd.DataFrame:
    """Baca seluruh riwayat (atau satu ``as_of``), satu baris per run."""
    keys = sink.list_keys(f"{dataset.GOLD_RUN_LOG}/")
    if as_of is not None:
        keys = [key for key in keys if f"as_of={as_of}/" in key]
    # Urutkan agar format lama (as_of=<D>/data.parquet) terbaca lebih dulu,
    # sehingga dedupe keep="last" memilih baris format baru.
    keys = sorted(keys)

    frames: list[pd.DataFrame] = []
    for key in keys:
        try:
            frames.append(sink.read_parquet(key))
        except Exception:  # noqa: BLE001 - file rusak tidak boleh mematikan laporan
            continue

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["as_of", "stage", "run_id"], keep="last")
    return combined.sort_values(["as_of", "stage", "run_id"]).reset_index(drop=True)
