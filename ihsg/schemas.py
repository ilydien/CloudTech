"""Kontrak kolom untuk setiap tabel medallion.

Kontrak adalah satu-satunya sumber kebenaran skema: dipakai untuk memvalidasi
sebelum menulis, dan didokumentasikan di ``docs/contracts.md``.

Sengaja tidak memakai pandera: satu dependensi lebih sedikit, dan pesan error
bisa disesuaikan dengan konteks pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Tipe logis -> kategori pandas yang diterima
_DTYPE_FAMILY = {
    "datetime": ("datetime",),
    "float": ("float", "integer"),
    "int": ("integer",),
    "str": ("string", "object"),
    "bool": ("bool",),
}


class ContractError(RuntimeError):
    """Data tidak memenuhi kontrak; pipeline harus berhenti."""


@dataclass(frozen=True)
class Contract:
    name: str
    columns: dict[str, str]
    keys: tuple[str, ...] = ()
    not_null: tuple[str, ...] = ()
    description: str = ""
    extra: tuple[str, ...] = field(default_factory=tuple)

    def expected_columns(self) -> tuple[str, ...]:
        return tuple(self.columns) + self.extra


def _family(series: pd.Series) -> str:
    dtype = series.dtype
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    if pd.api.types.is_integer_dtype(dtype):
        return "integer"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if pd.api.types.is_string_dtype(dtype):
        return "string"
    return "object"


def validate_contract(df: pd.DataFrame, contract: Contract) -> None:
    """Validasi skema. Raise :class:`ContractError` bila gagal."""
    problems: list[str] = []

    missing = [col for col in contract.expected_columns() if col not in df.columns]
    if missing:
        raise ContractError(f"{contract.name}: kolom hilang: {missing}")

    for column, logical in contract.columns.items():
        accepted = _DTYPE_FAMILY[logical]
        actual = _family(df[column])
        if actual not in accepted:
            problems.append(f"kolom {column!r} bertipe {actual}, diharapkan {logical}")

    for column in contract.not_null:
        if df[column].isna().any():
            problems.append(f"kolom {column!r} memiliki {int(df[column].isna().sum())} nilai null")

    if contract.keys and not df.empty:
        duplicated = df.duplicated(subset=list(contract.keys), keep=False)
        if duplicated.any():
            sample = df.loc[duplicated, list(contract.keys)].head(3).to_dict("records")
            problems.append(f"duplikat natural key {contract.keys}: contoh {sample}")

    if problems:
        raise ContractError(f"{contract.name}: " + "; ".join(problems))


def fingerprint(df: pd.DataFrame) -> str:
    """SHA-256 isi DataFrame (nilai + urutan), untuk lineage antar-layer."""
    import hashlib

    if df.empty:
        return hashlib.sha256(b"empty").hexdigest()
    values = pd.util.hash_pandas_object(df, index=False).values.tobytes()
    return hashlib.sha256(values).hexdigest()


# ---------------------------------------------------------------------------
# BRONZE - nilai mentah dari provider, nama kolom apa adanya
# ---------------------------------------------------------------------------
BRONZE_PRICES = Contract(
    name="bronze.prices",
    description="OHLCV mentah dari yfinance (auto_adjust=False). Tidak ada rename kolom.",
    columns={
        "Date": "datetime",
        "Open": "float",
        "High": "float",
        "Low": "float",
        "Close": "float",
        "Adj Close": "float",
        "Volume": "float",
    },
    keys=("Date",),
    not_null=("Date", "Close"),
)

BRONZE_ACTIONS = Contract(
    name="bronze.actions",
    description="Dividen & stock split mentah dari yfinance.",
    columns={"Date": "datetime", "Dividends": "float", "Stock Splits": "float"},
    keys=("Date",),
    not_null=("Date",),
)

BRONZE_MANIFEST = Contract(
    name="bronze.manifest",
    description="Catatan parameter unduhan + fingerprint per ticker (tanpa JSON).",
    columns={
        "ingest_date": "str",
        "ticker": "str",
        "ticker_key": "str",
        "rows": "int",
        "date_min": "str",
        "date_max": "str",
        "fingerprint": "str",
        "provider": "str",
        "provider_version": "str",
        "request_start": "str",
        "request_end": "str",
    },
    keys=("ticker",),
    not_null=("ticker", "fingerprint"),
)

# ---------------------------------------------------------------------------
# SILVER - bersih, bertipe, satu resolusi (harian)
# ---------------------------------------------------------------------------
SILVER_PRICES = Contract(
    name="silver.prices",
    description="Harga per ticker: dedup, tipe benar, return & Parkinson variance.",
    columns={
        "date": "datetime",
        "ticker": "str",
        "open": "float",
        "high": "float",
        "low": "float",
        "close": "float",
        "adj_close": "float",
        "volume": "float",
        "ret_cc_pct": "float",
        "park_var": "float",
        "source_ingest_date": "str",
    },
    keys=("ticker", "date"),
    not_null=("date", "close", "adj_close"),
)

SILVER_PANEL = Contract(
    name="silver.panel",
    description=(
        "Panel lintas instrumen pada kalender bursa BEI, ffill terbatas. "
        "Kolom: {KEY}_open/high/low/close/adj_close/volume/ret/park + flag is_filled_{KEY}."
    ),
    columns={"date": "datetime"},
    keys=("date",),
    not_null=("date",),
    extra=("JKSE_close",),
)

# ---------------------------------------------------------------------------
# GOLD - fitur siap model, prediksi, metrik, riwayat
# ---------------------------------------------------------------------------
GOLD_FEATURES = Contract(
    name="gold.features",
    description="Matriks fitur ber-lag + target IHSG. Baris terakhir = hari depan (target null).",
    columns={
        "date": "datetime",
        "JKSE_ret_lag1": "float",
        "JKSE_park_var_lag1": "float",
        "ret": "float",
        "park": "float",
        "close": "float",
        "prev_close": "float",
    },
    keys=("date",),
)

GOLD_PREDICTIONS = Contract(
    name="gold.predictions",
    description="Ramalan 1..H hari. Sumber JSON response Azure Function.",
    columns={
        "as_of": "str",
        "run_id": "str",
        "model_vol": "str",
        "model_ret": "str",
        "base_date": "str",
        "last_close": "float",
        "horizon_step": "int",
        "target_date": "str",
        "forecast_variance_pct2": "float",
        "forecast_volatility_pct": "float",
        "cumulative_variance_pct2": "float",
        "annualized_volatility_pct": "float",
        "risk_regime": "str",
        "prob_high": "float",
        "tau": "float",
        "persistence": "float",
        "ml_return_enabled": "bool",
        "expected_log_return_pct": "float",
        "cumulative_expected_return_pct": "float",
        "median_close_price": "float",
        "price_interval_low": "float",
        "price_interval_high": "float",
        "prob_up_vs_last_close": "float",
        "direction_signal": "str",
    },
    keys=("as_of", "horizon_step"),
    not_null=("as_of", "horizon_step", "target_date"),
)

GOLD_CANDIDATES = Contract(
    name="gold.candidates",
    description=(
        "Semua kandidat model volatilitas pada satu run: QLIKE validasi/uji dan "
        "ramalan h=1 untuk sesi berikutnya. predict() hanya membaca tabel ini, "
        "sehingga tidak ada lagi train/serve skew."
    ),
    columns={
        "as_of": "str",
        "run_id": "str",
        "model": "str",
        "val_qlike": "float",
        "test_qlike": "float",
        "h1_var": "float",
        "uncond": "float",
        "persistence": "float",
        "prob_high": "float",
        "selected": "bool",
    },
    keys=("as_of", "run_id", "model"),
    not_null=("as_of", "run_id", "model"),
)

GOLD_PUBLISHED = Contract(
    name="gold.published",
    description="Penanda bahwa prediksi untuk as_of sudah lengkap dan aman disajikan.",
    columns={
        "as_of": "str",
        "run_id": "str",
        "model_vol": "str",
        "horizons": "int",
        "ok": "bool",
    },
    keys=("as_of",),
    not_null=("as_of", "run_id"),
)

GOLD_REGIME = Contract(
    name="gold.regime",
    description="Probabilitas rezim HIGH per horizon (keluaran klasifikasi model).",
    columns={
        "as_of": "str",
        "run_id": "str",
        "horizon_step": "int",
        "target_date": "str",
        "prob_high": "float",
        "risk_regime": "str",
    },
    keys=("as_of", "horizon_step"),
    not_null=("as_of", "horizon_step"),
)

GOLD_REGIME_EVAL = Contract(
    name="gold.regime_eval",
    description="Klasifikasi rezim pada periode uji: probabilitas vs aktual. Bahan dashboard.",
    columns={
        "as_of": "str",
        "run_id": "str",
        "date": "datetime",
        "prob_high": "float",
        "predicted_high": "bool",
        "actual_high": "bool",
        "correct": "bool",
    },
    keys=("as_of", "date"),
)

GOLD_VOL_EVAL = Contract(
    name="gold.vol_eval",
    description="Varians aktual vs ramalan beberapa model pada periode uji. Bahan dashboard.",
    columns={
        "as_of": "str",
        "run_id": "str",
        "date": "datetime",
        "actual_var": "float",
        "gru_var": "float",
        "har_var": "float",
        "ewma_var": "float",
        "hist_var": "float",
    },
    keys=("as_of", "date"),
)

GOLD_METRICS = Contract(
    name="gold.metrics",
    description="Metrik panjang: model x metric x horizon x split.",
    columns={
        "as_of": "str",
        "run_id": "str",
        "split": "str",
        "model": "str",
        "metric": "str",
        "horizon": "int",
        "value": "float",
    },
    keys=("as_of", "run_id", "split", "model", "metric", "horizon"),
)

GOLD_RUN_LOG = Contract(
    name="gold.run_log",
    description="Riwayat run per tahap. Bukti 'retrain > 3x' tanpa bergantung pada UI orkestrator.",
    columns={
        "as_of": "str",
        "run_id": "str",
        "trigger": "str",
        "stage": "str",
        "status": "str",
        "promoted": "bool",
        "val_metric": "float",
        "test_metric": "float",
        "duration_s": "float",
        "data_fingerprint": "str",
        "config_hash": "str",
        "git_sha": "str",
    },
    keys=("as_of", "stage", "run_id"),
)

GOLD_PARAMS = Contract(
    name="gold.params",
    description="Parameter turunan split: c_train, tau, long-run variance, batas indeks.",
    columns={
        "as_of": "str",
        "c_train": "float",
        "tau": "float",
        "long_run_var_pct2": "float",
        "persistence": "float",
        "n_train": "int",
        "n_val": "int",
        "n_test": "int",
        "i_tr": "int",
        "i_va": "int",
        "n_real": "int",
        "n_features": "int",
        "window": "int",
        "data_fingerprint": "str",
    },
    keys=("as_of",),
)

GOLD_CHAMPION = Contract(
    name="gold.champion",
    description="Model yang sedang dipakai produksi (satu baris).",
    columns={
        "run_id": "str",
        "trained_as_of": "str",
        "model_vol": "str",
        "model_ret": "str",
        "val_qlike": "float",
        "val_r2_oos": "float",
        "ml_return_enabled": "bool",
        "tau": "float",
        "c_train": "float",
        "long_run_var_pct2": "float",
        "persistence": "float",
        "window": "int",
        "horizons": "int",
        "n_features": "int",
        "config_hash": "str",
        "data_fingerprint": "str",
    },
    keys=("run_id",),
    not_null=("run_id",),
)

# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------
DQ_REPORT = Contract(
    name="dq.report",
    description="Hasil pemeriksaan kualitas data. details = 'k=v;k=v' (bukan JSON).",
    columns={
        "as_of": "str",
        "layer": "str",
        "table": "str",
        "check_name": "str",
        "severity": "str",
        "passed": "bool",
        "failed_count": "int",
        "total_count": "int",
        "details": "str",
    },
    keys=("as_of", "layer", "table", "check_name"),
)

QUARANTINE = Contract(
    name="dq.quarantine",
    description="Baris yang tidak dipakai, beserta alasannya.",
    columns={
        "as_of": "str",
        "layer": "str",
        "table": "str",
        "reason": "str",
        "key_value": "str",
        "column_name": "str",
        "column_value": "str",
    },
    keys=(),
)
