"""Antarmuka baris perintah.

Semua tahapan punya satu titik masuk:

    python -m ihsg.cli ingest    --as-of 2026-11-13
    python -m ihsg.cli silver    --as-of 2026-11-13
    python -m ihsg.cli features  --as-of 2026-11-13
    python -m ihsg.cli train     --as-of 2026-11-13 --trigger manual
    python -m ihsg.cli predict   --as-of 2026-11-13
    python -m ihsg.cli run       --as-of auto [--retrain] [--trigger schedule]
    python -m ihsg.cli backfill  --start 2026-10-01 --end 2026-10-31
    python -m ihsg.cli verify    [--as-of 2026-11-13]
    python -m ihsg.cli report
    python -m ihsg.cli doctor

Semua perintah mencetak hasil sebagai JSON ke stdout (kecuali ``report`` yang
mencetak tabel). Exit code != 0 berarti gagal - ini yang dipakai cron/Prefect.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import traceback

import pandas as pd

from ihsg import dataset
from ihsg.config import Settings
from ihsg.pipelines import features as features_pipeline
from ihsg.pipelines import ingest as ingest_pipeline
from ihsg.pipelines import predict as predict_pipeline
from ihsg.pipelines import silver as silver_pipeline
from ihsg.pipelines import train as train_pipeline
from ihsg.storage import get_sink


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    if getattr(args, "sink", None):
        settings = dataclasses.replace(settings, sink=args.sink)
    return settings


def _resolve(settings: Settings, as_of: str | None) -> str:
    if as_of in (None, "", "auto"):
        return settings.as_of or dataset.resolve_as_of(settings)
    return as_of


def _print(payload: dict | list) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


# ---------------------------------------------------------------------------
# Perintah
# ---------------------------------------------------------------------------
def cmd_ingest(args: argparse.Namespace) -> dict:
    settings = _settings(args)
    return ingest_pipeline.run(settings, _resolve(settings, args.as_of), force=args.force)


def cmd_silver(args: argparse.Namespace) -> dict:
    settings = _settings(args)
    return silver_pipeline.run(settings, _resolve(settings, args.as_of))


def cmd_features(args: argparse.Namespace) -> dict:
    settings = _settings(args)
    return features_pipeline.run(settings, _resolve(settings, args.as_of))


def cmd_train(args: argparse.Namespace) -> dict:
    settings = _settings(args)
    return train_pipeline.run(settings, _resolve(settings, args.as_of), trigger=args.trigger)


def cmd_predict(args: argparse.Namespace) -> dict:
    settings = _settings(args)
    return predict_pipeline.run(settings, _resolve(settings, args.as_of))


def cmd_run(args: argparse.Namespace) -> dict:
    settings = _settings(args)
    requested_as_of = _resolve(settings, args.as_of)
    ingest_result = ingest_pipeline.run(settings, requested_as_of)
    # as_of hasil ingest bisa lebih awal dari kalender (provider tertinggal);
    # seluruh tahap berikutnya WAJIB memakai tanggal yang sama.
    as_of = ingest_result["as_of"]
    stages: list[dict] = [
        ingest_result,
        silver_pipeline.run(settings, as_of),
        features_pipeline.run(settings, as_of),
    ]
    if args.retrain:
        stages.append(train_pipeline.run(settings, as_of, trigger=args.trigger))
    stages.append(predict_pipeline.run(settings, as_of))
    return {"stage": "run", "as_of": as_of, "stages": stages}


def cmd_backfill(args: argparse.Namespace) -> dict:
    settings = _settings(args)
    start = args.start or settings.start_date
    end = args.end or dataset.resolve_as_of(settings)
    results: list[dict] = []

    for day in dataset.trading_days(start, end, settings.idx_holidays).date:
        as_of = day.isoformat()
        record: dict = {"requested_as_of": as_of}
        try:
            ingest_result = ingest_pipeline.run(settings, as_of)
            resolved = ingest_result["as_of"]
            record["as_of"] = resolved
            silver_pipeline.run(settings, resolved)
            features_pipeline.run(settings, resolved)
            if args.retrain:
                record["train"] = train_pipeline.run(settings, resolved, trigger="backfill")
            record["predict"] = predict_pipeline.run(settings, resolved)
            record["status"] = "ok"
        except Exception as error:  # noqa: BLE001 - backfill harus lanjut
            record["status"] = "failed"
            record["error"] = f"{type(error).__name__}: {error}"
        results.append(record)
        print(f"[backfill] {as_of} -> {record['status']}", file=sys.stderr)

    return {
        "stage": "backfill",
        "start": start,
        "end": end,
        "days": len(results),
        "ok": sum(1 for item in results if item["status"] == "ok"),
        "failed": sum(1 for item in results if item["status"] == "failed"),
        "results": results,
    }


def cmd_verify(args: argparse.Namespace) -> dict:
    settings = _settings(args)
    sink = get_sink(settings)
    from ihsg.serving import handler

    as_of = args.as_of or (dataset.existing_as_of(sink, dataset.GOLD_PREDICTIONS) or [None])[-1]
    if as_of is None:
        raise SystemExit("belum ada gold/predictions; jalankan prediksi dulu")
    payload = handler.build_payload(settings, as_of)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    return {
        "stage": "verify",
        "as_of": as_of,
        "payload_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "horizons": len(payload["forecast_horizons"]),
        "model": payload["model"]["volatility"],
    }


def cmd_report(args: argparse.Namespace) -> dict:
    settings = _settings(args)
    sink = get_sink(settings)
    frames = []
    for as_of in dataset.existing_as_of(sink, dataset.GOLD_RUN_LOG):
        frames.append(sink.read_parquet(dataset.as_of_key(dataset.GOLD_RUN_LOG, as_of)))
    if not frames:
        print("belum ada run tercatat")
        return {"stage": "report", "runs": 0}
    table = pd.concat(frames, ignore_index=True).sort_values("as_of")
    columns = ["as_of", "run_id", "trigger", "promoted", "val_metric", "test_metric", "duration_s"]
    print(table[columns].to_string(index=False))
    return {"stage": "report", "runs": int(len(table))}


def cmd_doctor(args: argparse.Namespace) -> dict:
    settings = _settings(args)
    sink = get_sink(settings)
    probe_key = "gold/_doctor/probe.parquet"
    sink.write_parquet(pd.DataFrame({"ok": [1]}), probe_key, overwrite=True)
    readable = len(sink.read_parquet(probe_key)) == 1
    sink.delete(probe_key)

    prefixes = {
        "bronze": len(sink.list_keys("bronze/")),
        "silver": len(sink.list_keys("silver/")),
        "gold": len(sink.list_keys("gold/")),
        "mlartifacts": len(sink.list_keys("mlartifacts/")),
    }
    return {
        "stage": "doctor",
        "sink": settings.sink,
        "endpoint": settings.s3_endpoint_url or settings.local_root,
        "write_read_ok": bool(readable),
        "objects": prefixes,
        "config_hash": settings.config_hash(),
        "as_of_auto": dataset.resolve_as_of(settings),
    }


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ihsg", description="IHSG forecasting engine")
    parser.add_argument("--version", action="version", version="ihsg 4.2.0")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--as-of", default=None, help="YYYY-MM-DD atau 'auto'")
        target.add_argument("--sink", choices=["local", "s3"], default=None)

    ingest = sub.add_parser("ingest", help="unduh bronze (append-only)")
    common(ingest)
    ingest.add_argument("--force", action="store_true")
    ingest.set_defaults(func=cmd_ingest)

    for name, func, help_text in (
        ("silver", cmd_silver, "bangun silver dari bronze"),
        ("features", cmd_features, "bangun gold/features"),
        ("predict", cmd_predict, "ramalan 1..H hari"),
    ):
        command = sub.add_parser(name, help=help_text)
        common(command)
        command.set_defaults(func=func)

    train = sub.add_parser("train", help="latih GRU + evaluasi + gate promosi")
    common(train)
    train.add_argument("--trigger", default="manual", choices=["manual", "schedule", "backfill", "api"])
    train.set_defaults(func=cmd_train)

    run = sub.add_parser("run", help="ingest -> silver -> features -> [train] -> predict")
    common(run)
    run.add_argument("--retrain", action="store_true")
    run.add_argument("--trigger", default="schedule", choices=["manual", "schedule", "backfill", "api"])
    run.set_defaults(func=cmd_run)

    backfill = sub.add_parser("backfill", help="jalankan pipeline untuk rentang tanggal")
    common(backfill)
    backfill.add_argument("--start", default=None)
    backfill.add_argument("--end", default=None)
    backfill.add_argument("--retrain", action="store_true")
    backfill.set_defaults(func=cmd_backfill)

    verify = sub.add_parser("verify", help="bangun payload dan cetak hash-nya")
    verify.add_argument("--as-of", default=None)
    verify.add_argument("--sink", choices=["local", "s3"], default=None)
    verify.set_defaults(func=cmd_verify)

    report = sub.add_parser("report", help="tabel riwayat run/retrain")
    report.add_argument("--sink", choices=["local", "s3"], default=None)
    report.set_defaults(func=cmd_report)

    doctor = sub.add_parser("doctor", help="cek konektivitas storage")
    doctor.add_argument("--sink", choices=["local", "s3"], default=None)
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
    except Exception as error:  # noqa: BLE001 - CLI harus memberi exit code jelas
        traceback.print_exc()
        print(json.dumps({"status": "error", "error": f"{type(error).__name__}: {error}"}), file=sys.stderr)
        return 1
    if result is not None:
        _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
