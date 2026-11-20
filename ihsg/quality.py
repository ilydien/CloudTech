"""Pemeriksaan kualitas data: hard-fail, soft-fail, dan karantina.

Aturan main:
* **hard** - melanggar berarti pipeline berhenti (skema, duplikat natural key,
  tanggal acuan tidak ada). Tidak boleh dilewati.
* **soft** - dicatat di ``silver/_dq`` dan dilaporkan, tidak menghentikan run
  (kelengkapan, gap, freshness, proporsi volume nol, outlier).
* **quarantine** - baris yang tidak dipakai ditulis ke ``silver/_quarantine``
  beserta alasannya, supaya bisa diperiksa tanpa mengubah bronze.

``details`` disimpan sebagai string ``k=v;k=v`` - bukan JSON, sesuai batasan
media penyimpanan proyek.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ihsg import dataset
from ihsg.schemas import DQ_REPORT, QUARANTINE, validate_contract
from ihsg.storage import Sink

HARD = "hard"
SOFT = "soft"
INFO = "info"


class QualityError(RuntimeError):
    """Pelanggaran aturan hard."""


@dataclass
class CheckResult:
    layer: str
    table: str
    check_name: str
    severity: str
    failed_count: int
    total_count: int
    details: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.failed_count == 0

    def as_row(self, as_of: str) -> dict:
        detail_text = ";".join(f"{key}={value}" for key, value in sorted(self.details.items()))
        return {
            "as_of": as_of,
            "layer": self.layer,
            "table": self.table,
            "check_name": self.check_name,
            "severity": self.severity,
            "passed": bool(self.passed),
            "failed_count": int(self.failed_count),
            "total_count": int(self.total_count),
            "details": detail_text,
        }

    def raise_if_hard(self) -> None:
        if self.severity == HARD and not self.passed:
            raise QualityError(
                f"[{self.table}] {self.check_name} gagal: {self.failed_count}/{self.total_count} "
                f"({self.details})"
            )


# ---------------------------------------------------------------------------
# Pemeriksaan individual
# ---------------------------------------------------------------------------
def check_not_null(df: pd.DataFrame, table: str, layer: str, columns: list[str]) -> CheckResult:
    failed = int(df[columns].isna().any(axis=1).sum()) if len(df) else 0
    return CheckResult(
        layer=layer,
        table=table,
        check_name="not_null",
        severity=HARD,
        failed_count=failed,
        total_count=len(df),
        details={"columns": ",".join(columns)},
    )


def check_positive(
    df: pd.DataFrame,
    table: str,
    layer: str,
    column: str,
    severity: str = HARD,
) -> CheckResult:
    """Harga harus positif.

    Default HARD, tapi untuk futures bisa dilonggarkan: WTI pernah ditutup
    negatif (-37,63 pada 2020-04-20) dan itu data pasar yang sah, bukan cacat.
    """
    invalid = ~(df[column] > 0)
    return CheckResult(
        layer=layer,
        table=table,
        check_name=f"positive_{column}",
        severity=severity,
        failed_count=int(invalid.sum()) if len(df) else 0,
        total_count=len(df),
        details={"column": column},
    )


def check_range(
    df: pd.DataFrame,
    table: str,
    layer: str,
    column: str,
    low: float,
    high: float,
) -> CheckResult:
    outside = (df[column] < low) | (df[column] > high)
    outside &= df[column].notna()
    return CheckResult(
        layer=layer,
        table=table,
        check_name=f"range_{column}",
        severity=SOFT,
        failed_count=int(outside.sum()) if len(df) else 0,
        total_count=len(df),
        details={"low": low, "high": high},
    )


def check_calendar_gaps(
    df: pd.DataFrame,
    table: str,
    layer: str,
    holidays: tuple[str, ...],
) -> CheckResult:
    """Hari bursa yang hilang antara tanggal pertama dan terakhir."""
    if len(df) < 2:
        return CheckResult(layer, table, "calendar_gaps", SOFT, 0, len(df), {})
    first = pd.Timestamp(df["date"].min()).date().isoformat()
    last = pd.Timestamp(df["date"].max()).date().isoformat()
    expected = set(dataset.trading_days(first, last, holidays).date)
    actual = set(pd.to_datetime(df["date"]).dt.date)
    missing = sorted(expected - actual)
    return CheckResult(
        layer=layer,
        table=table,
        check_name="calendar_gaps",
        severity=SOFT,
        failed_count=len(missing),
        total_count=len(expected),
        details={"first_missing": missing[0] if missing else "-", "last_missing": missing[-1] if missing else "-"},
    )


def check_freshness(
    df: pd.DataFrame,
    table: str,
    layer: str,
    as_of: str,
    column: str = "date",
    max_lag_days: int = 3,
) -> CheckResult:
    """Data harus mencapai ``as_of`` (toleransi ``max_lag_days``)."""
    if df.empty:
        return CheckResult(layer, table, "freshness", HARD, 1, 1, {"as_of": as_of})
    latest = pd.Timestamp(df[column].max()).date()
    lag = (pd.Timestamp(as_of).date() - latest).days
    failed = 1 if lag > max_lag_days else 0
    return CheckResult(
        layer=layer,
        table=table,
        check_name="freshness",
        severity=HARD,
        failed_count=failed,
        total_count=1,
        details={"as_of": as_of, "latest": latest.isoformat(), "lag_days": lag},
    )


def check_zero_volume_share(
    df: pd.DataFrame, table: str, layer: str, column: str = "volume", threshold: float = 0.9
) -> CheckResult:
    """Instrumen seperti kurs memang bervolume nol: catat, jangan gagalkan."""
    share = float((df[column].fillna(0) <= 0).mean()) if len(df) else 0.0
    failed = 1 if share >= threshold else 0
    return CheckResult(
        layer=layer,
        table=table,
        check_name="zero_volume_share",
        severity=SOFT,
        failed_count=failed,
        total_count=len(df),
        details={"share": round(share, 4)},
    )


def check_outliers(
    df: pd.DataFrame,
    table: str,
    layer: str,
    column: str,
    z_threshold: float = 10.0,
) -> CheckResult:
    values = df[column].astype(float)
    sigma = float(values.std(ddof=0))
    if not np.isfinite(sigma) or sigma == 0.0:
        return CheckResult(layer, table, f"outlier_{column}", INFO, 0, len(df), {})
    z = (values - values.mean()).abs() / sigma
    outside = z > z_threshold
    return CheckResult(
        layer=layer,
        table=table,
        check_name=f"outlier_{column}",
        severity=INFO,
        failed_count=int(outside.sum()),
        total_count=len(df),
        details={"z": z_threshold},
    )


# ---------------------------------------------------------------------------
# Penulisan laporan & karantina
# ---------------------------------------------------------------------------
def report_frame(results: list[CheckResult], as_of: str) -> pd.DataFrame:
    frame = pd.DataFrame([result.as_row(as_of) for result in results])
    validate_contract(frame, DQ_REPORT)
    return frame


def write_report(sink: Sink, results: list[CheckResult], as_of: str, layer: str = "silver") -> str | None:
    if not results:
        return None
    prefix = dataset.SILVER_DQ if layer == "silver" else "gold/_dq"
    key = dataset.as_of_key(prefix, as_of)
    sink.write_parquet(report_frame(results, as_of), key, overwrite=True)
    return key


def quarantine_frame(
    df: pd.DataFrame,
    mask: pd.Series,
    reason: str,
    layer: str,
    table: str,
    as_of: str,
    key_column: str,
    value_column: str,
) -> pd.DataFrame:
    """Ubah baris yang gagal menjadi tabel karantina (satu baris per kejadian).

    ``key_value``   = nilai kunci baris (mis. tanggal), sebagai string
    ``column_name`` = NAMA kolom yang bermasalah
    ``column_value``= nilai kolom tersebut, sebagai string
    """
    columns = ["as_of", "layer", "table", "reason", "key_value", "column_name", "column_value"]
    if not mask.any():
        return pd.DataFrame(columns=columns)

    selected = pd.DataFrame(
        {
            "as_of": as_of,
            "layer": layer,
            "table": table,
            "reason": reason,
            "key_value": df.loc[mask, key_column].astype(str).to_numpy(),
            "column_name": value_column,
            "column_value": df.loc[mask, value_column].astype(str).to_numpy(),
        }
    )
    return selected[columns]


def write_quarantine(sink: Sink, frame: pd.DataFrame, as_of: str, layer: str = "silver") -> str | None:
    if frame is None or frame.empty:
        return None
    validate_contract(frame, QUARANTINE)
    prefix = "silver/_quarantine" if layer == "silver" else "gold/_quarantine"
    key = dataset.as_of_key(prefix, as_of)
    sink.write_parquet(frame, key, overwrite=True)
    return key
