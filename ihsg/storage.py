"""Abstraksi penyimpanan (medallion) untuk Parquet dan artefak model.

Satu-satunya tempat yang tahu apakah data ada di filesystem atau di object
storage S3-compatible (Garage/MinIO/Azure Blob via gateway S3).

Key selalu berbentuk ``<bucket>/<path>/data.parquet``:

    bronze/yf/prices/ticker=JKSE/ingest_date=2026-11-13/data.parquet
    gold/predictions/as_of=2026-11-13/data.parquet
    mlartifacts/run_id=abc123/model.pt

Segmen pertama adalah bucket (bronze/silver/gold/mlartifacts), sisanya path di
dalam bucket. Dengan begitu satu key namespace berlaku untuk lokal maupun S3,
dan pindah ke Azure cukup dengan mengimplementasikan ``Sink`` baru.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ihsg.config import Settings


class SinkError(RuntimeError):
    """Kesalahan penyimpanan dengan pesan yang menyebut key-nya."""


class Sink(ABC):
    """Kontrak minimum yang dibutuhkan pipeline."""

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def read_parquet(self, key: str) -> pd.DataFrame: ...

    @abstractmethod
    def write_parquet(self, df: pd.DataFrame, key: str, *, overwrite: bool = True) -> bool:
        """Tulis DataFrame sebagai Parquet. Return False bila di-skip."""

    @abstractmethod
    def write_bytes(self, data: bytes, key: str, *, overwrite: bool = True) -> bool: ...

    @abstractmethod
    def read_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def list_keys(self, prefix: str) -> list[str]: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    # -- helper ------------------------------------------------------------
    def try_read_parquet(self, key: str) -> pd.DataFrame | None:
        return self.read_parquet(key) if self.exists(key) else None

    @staticmethod
    def _to_table(df: pd.DataFrame) -> pa.Table:
        # Index tidak pernah disimpan: semua tanggal adalah kolom biasa.
        return pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False)


def _split_key(key: str) -> tuple[str, str]:
    bucket, _, rest = key.partition("/")
    if not bucket:
        raise SinkError(f"key tidak valid (bucket kosong): {key!r}")
    return bucket, rest


class LocalSink(Sink):
    """Penyimpanan lokal: ``<root>/<bucket>/<path>``. Untuk debugging cepat."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def read_parquet(self, key: str) -> pd.DataFrame:
        path = self._path(key)
        if not path.is_file():
            raise SinkError(f"objek tidak ditemukan: {key}")
        # Baca lewat file handle, BUKAN lewat path string: pyarrow 18 menganggap
        # path yang mengandung "k=v" sebagai dataset Hive dan menambahkan kolom
        # partisi secara diam-diam (mis. `as_of`), sehingga skema berubah.
        with path.open("rb") as handle:
            return pq.read_table(handle).to_pandas()

    def write_parquet(self, df: pd.DataFrame, key: str, *, overwrite: bool = True) -> bool:
        path = self._path(key)
        if path.exists() and not overwrite:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(self._to_table(df), tmp, compression="snappy")
        os.replace(tmp, path)
        return True

    def write_bytes(self, data: bytes, key: str, *, overwrite: bool = True) -> bool:
        path = self._path(key)
        if path.exists() and not overwrite:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        return True

    def read_bytes(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise SinkError(f"objek tidak ditemukan: {key}")
        return path.read_bytes()

    def list_keys(self, prefix: str) -> list[str]:
        prefix = prefix.rstrip("/")
        base = self._path(prefix)
        if base.is_dir():
            files = [p for p in base.rglob("*") if p.is_file()]
        elif base.parent.is_dir():
            files = [p for p in base.parent.glob(Path(prefix).name + "*") if p.is_file()]
        else:
            return []
        return sorted(p.relative_to(self.root).as_posix() for p in files)

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()


class S3Sink(Sink):
    """S3-compatible: Garage, MinIO, atau Azure Blob lewat gateway S3."""

    def __init__(self, endpoint_url: str, region: str, access_key: str, secret_key: str) -> None:
        import s3fs  # impor lazy: modul ini tetap bisa dipakai tanpa s3fs

        self._fs = s3fs.S3FileSystem(
            key=access_key,
            secret=secret_key,
            use_ssl=endpoint_url.startswith("https://"),
            client_kwargs={"endpoint_url": endpoint_url, "region_name": region},
            config_kwargs={"retries": {"max_attempts": 5, "mode": "standard"}},
        )

    @staticmethod
    def _full(key: str) -> str:
        bucket, rest = _split_key(key)
        return f"{bucket}/{rest}" if rest else bucket

    def exists(self, key: str) -> bool:
        return bool(self._fs.exists(self._full(key)))

    def read_parquet(self, key: str) -> pd.DataFrame:
        full = self._full(key)
        if not self._fs.exists(full):
            raise SinkError(f"objek tidak ditemukan: {key}")
        with self._fs.open(full, "rb") as handle:
            return pq.read_table(handle).to_pandas()

    def write_parquet(self, df: pd.DataFrame, key: str, *, overwrite: bool = True) -> bool:
        full = self._full(key)
        if not overwrite and self._fs.exists(full):
            return False
        buffer = pa.BufferOutputStream()
        pq.write_table(self._to_table(df), buffer, compression="snappy")
        with self._fs.open(full, "wb") as handle:
            handle.write(buffer.getvalue().to_pybytes())
        return True

    def write_bytes(self, data: bytes, key: str, *, overwrite: bool = True) -> bool:
        full = self._full(key)
        if not overwrite and self._fs.exists(full):
            return False
        with self._fs.open(full, "wb") as handle:
            handle.write(data)
        return True

    def read_bytes(self, key: str) -> bytes:
        full = self._full(key)
        if not self._fs.exists(full):
            raise SinkError(f"objek tidak ditemukan: {key}")
        with self._fs.open(full, "rb") as handle:
            return handle.read()

    def list_keys(self, prefix: str) -> list[str]:
        prefix = prefix.rstrip("/")
        bucket, rest = _split_key(prefix)
        full_prefix = f"{bucket}/{rest}" if rest else bucket
        try:
            found = self._fs.find(full_prefix)
        except FileNotFoundError:
            return []
        return sorted(path for path in found if path.startswith(full_prefix))

    def delete(self, key: str) -> None:
        full = self._full(key)
        if self._fs.exists(full):
            self._fs.rm(full)


def _build_sink(kind: str, endpoint: str, region: str, access: str, secret: str) -> Sink:
    if kind == "local":
        return LocalSink(endpoint)  # pada mode local, "endpoint" = root path
    return S3Sink(endpoint_url=endpoint, region=region, access_key=access, secret_key=secret)


@lru_cache(maxsize=8)
def _cached_sink(kind: str, endpoint: str, region: str, access: str, secret: str) -> Sink:
    return _build_sink(kind, endpoint, region, access, secret)


def get_sink(settings: Settings) -> Sink:
    """Sink sesuai ``SINK`` di environment (di-cache per konfigurasi)."""
    if settings.sink == "local":
        return _cached_sink("local", settings.local_root, "-", "-", "-")
    if not settings.s3_endpoint_url or not settings.s3_access_key or not settings.s3_secret_key:
        raise SinkError(
            "SINK=s3 membutuhkan S3_ENDPOINT_URL, S3_ACCESS_KEY (atau GARAGE_DEFAULT_ACCESS_KEY), "
            "dan S3_SECRET_KEY"
        )
    return _cached_sink(
        "s3",
        settings.s3_endpoint_url,
        settings.s3_region,
        settings.s3_access_key,
        settings.s3_secret_key,
    )
