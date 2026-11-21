"""Konfigurasi terpusat, dibaca dari environment.

Tidak ada file konfigurasi model: image yang sama harus bisa berjalan di
laptop, di Docker Compose, dan di Azure Container Apps Job hanya dengan
mengubah environment. Nilai yang memengaruhi keluaran model dikumpulkan di
``model_config()`` dan di-hash menjadi ``config_hash`` — hash ini ikut
disimpan di setiap artefak sehingga hasil bisa direproduksi.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, fields

JAKARTA_TZ = "Asia/Jakarta"

DEFAULT_TICKERS = "^JKSE,CL=F,GC=F,IDR=X,^HSI,^N225,^DJI,EEM"


# ---------------------------------------------------------------------------
# Pembacaan environment
# ---------------------------------------------------------------------------
def env(name: str, default: str | None = None) -> str | None:
    """Baca env; string kosong dianggap tidak di-set."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def env_int(name: str, default: int) -> int:
    value = env(name)
    return int(value) if value is not None else default


def env_float(name: str, default: float) -> float:
    value = env(name)
    return float(value) if value is not None else default


def env_bool(name: str, default: bool) -> bool:
    value = env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def env_list(name: str, default: str = "") -> tuple[str, ...]:
    raw = env(name, default) or ""
    return tuple(item.strip() for item in raw.split(",") if item.strip())


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """Seluruh knob pipeline dalam satu objek immutable."""

    # identitas run
    as_of: str | None
    start_date: str
    tickers: tuple[str, ...]
    idx_holidays: tuple[str, ...]

    # storage
    sink: str
    local_root: str
    s3_endpoint_url: str | None
    s3_region: str
    s3_access_key: str | None
    s3_secret_key: str | None

    # pemodelan
    seed: int
    window: int
    horizons: int
    refit_every: int
    train_frac: float
    val_frac: float
    tau_quantile: float
    ffill_limit: int
    hidden: int
    layers: int
    dropout: float
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    ridge_alpha: float
    ml_gate_alpha: float
    ewma_lambda: float

    # serving
    admin_token: str | None

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            as_of=env("AS_OF"),
            start_date=env("START_DATE", "2018-01-01") or "2018-01-01",
            tickers=env_list("TICKERS", DEFAULT_TICKERS),
            idx_holidays=env_list("IDX_HOLIDAYS"),
            sink=(env("SINK", "s3") or "s3").lower(),
            local_root=env("LOCAL_ROOT", "/app/state/data") or "/app/state/data",
            s3_endpoint_url=env("S3_ENDPOINT_URL"),
            s3_region=env("S3_REGION", "garage") or "garage",
            s3_access_key=env("S3_ACCESS_KEY") or env("GARAGE_DEFAULT_ACCESS_KEY"),
            s3_secret_key=env("S3_SECRET_KEY") or env("GARAGE_DEFAULT_SECRET_KEY"),
            seed=env_int("SEED", 42),
            window=env_int("WINDOW", 22),
            horizons=env_int("HORIZONS", 7),
            refit_every=env_int("REFIT_EVERY", 21),
            train_frac=env_float("TRAIN_FRAC", 0.60),
            val_frac=env_float("VAL_FRAC", 0.20),
            tau_quantile=env_float("TAU_QUANTILE", 0.95),
            ffill_limit=env_int("FFILL_LIMIT", 3),
            hidden=env_int("HIDDEN", 32),
            layers=env_int("LAYERS", 1),
            dropout=env_float("DROPOUT", 0.10),
            epochs=env_int("EPOCHS", 120),
            batch_size=env_int("BATCH_SIZE", 32),
            learning_rate=env_float("LEARNING_RATE", 0.003),
            weight_decay=env_float("WEIGHT_DECAY", 0.0001),
            patience=env_int("PATIENCE", 15),
            ridge_alpha=env_float("RIDGE_ALPHA", 1000.0),
            ml_gate_alpha=env_float("ML_GATE_ALPHA", 0.05),
            ewma_lambda=env_float("EWMA_LAMBDA", 0.94),
            admin_token=env("ADMIN_TOKEN"),
        )

    # -- hash konfigurasi ---------------------------------------------------
    @staticmethod
    def _model_fields() -> tuple[str, ...]:
        return (
            "seed", "window", "horizons", "refit_every", "train_frac", "val_frac",
            "tau_quantile", "ffill_limit", "hidden", "layers", "dropout", "epochs",
            "batch_size", "learning_rate", "weight_decay", "patience", "ridge_alpha",
            "ewma_lambda",
        )

    def model_config(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self._model_fields()}

    def config_hash(self) -> str:
        """Hash deterministik atas nilai yang memengaruhi keluaran model.

        Sengaja memakai gabungan ``k=v`` dan bukan JSON: hasilnya harus stabil
        lintas versi library dan tidak bergantung pada urutan dict.
        """
        payload = "|".join(f"{key}={self.model_config()[key]}" for key in sorted(self._model_fields()))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    # -- validasi -----------------------------------------------------------
    def validate(self) -> None:
        problems: list[str] = []
        if self.sink not in {"local", "s3"}:
            problems.append(f"SINK harus 'local' atau 's3', bukan {self.sink!r}")
        if not 0.1 <= self.train_frac <= 0.9:
            problems.append("TRAIN_FRAC harus di antara 0.1 dan 0.9")
        if not 0.0 <= self.val_frac <= 0.5:
            problems.append("VAL_FRAC harus di antara 0.0 dan 0.5")
        if self.train_frac + self.val_frac >= 0.95:
            problems.append("TRAIN_FRAC + VAL_FRAC harus < 0.95 (sisakan data uji)")
        if self.window < 5:
            problems.append("WINDOW minimal 5")
        if self.horizons < 1:
            problems.append("HORIZONS minimal 1")
        if self.sink == "s3" and not self.s3_endpoint_url:
            problems.append("SINK=s3 membutuhkan S3_ENDPOINT_URL")
        if problems:
            raise ValueError("Konfigurasi tidak valid:\n- " + "\n- ".join(problems))

    def describe(self) -> str:
        return (
            f"as_of={self.as_of or 'auto'} sink={self.sink} "
            f"window={self.window} horizons={self.horizons} "
            f"seed={self.seed} config_hash={self.config_hash()}"
        )


def field_names() -> list[str]:
    """Daftar nama field Settings (dipakai test)."""
    return [f.name for f in fields(Settings)]
