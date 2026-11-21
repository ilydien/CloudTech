from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from ihsg.config import Settings
from ihsg.storage import AzureBlobSink, LocalSink, SinkError, get_sink


@pytest.fixture
def sink(tmp_path) -> LocalSink:
    return LocalSink(tmp_path)


def test_parquet_roundtrip(sink: LocalSink):
    frame = pd.DataFrame({"date": [pd.Timestamp("2026-11-13")], "value": [1.5]})
    key = "gold/probe/as_of=2026-11-13/data.parquet"
    assert sink.write_parquet(frame, key) is True
    assert sink.exists(key)
    result = sink.read_parquet(key)
    pd.testing.assert_frame_equal(result, frame)


def test_no_overwrite_guard(sink: LocalSink):
    key = "bronze/yf/prices/ticker=JKSE/ingest_date=2026-11-13/data.parquet"
    first = pd.DataFrame({"Close": [1.0]})
    second = pd.DataFrame({"Close": [2.0]})
    assert sink.write_parquet(first, key, overwrite=False) is True
    assert sink.write_parquet(second, key, overwrite=False) is False
    assert float(sink.read_parquet(key)["Close"].iloc[0]) == 1.0


def test_bytes_roundtrip_for_model_artifacts(sink: LocalSink):
    key = "mlartifacts/run_id=abc/model.pt"
    payload = b"\x00\x01binary"
    assert sink.write_bytes(payload, key) is True
    assert sink.read_bytes(key) == payload


def test_list_keys_and_delete(sink: LocalSink):
    sink.write_parquet(pd.DataFrame({"x": [1]}), "silver/panel/data.parquet")
    sink.write_parquet(pd.DataFrame({"x": [2]}), "silver/prices/ticker=JKSE/data.parquet")
    keys = sink.list_keys("silver/")
    assert "silver/panel/data.parquet" in keys
    assert len(keys) == 2

    sink.delete("silver/panel/data.parquet")
    assert not sink.exists("silver/panel/data.parquet")
    assert len(sink.list_keys("silver/")) == 1


def test_missing_object_raises(sink: LocalSink):
    with pytest.raises(SinkError):
        sink.read_parquet("gold/predictions/as_of=2026-11-13/data.parquet")


def test_try_read_returns_none_when_absent(sink: LocalSink):
    assert sink.try_read_parquet("gold/champion/data.parquet") is None


def test_key_requires_bucket(sink: LocalSink):
    # Key tanpa bucket akan menulis ke luar struktur bucket; pastikan terdeteksi
    # saat membaca key yang tidak ada tetap menghasilkan error yang jelas.
    with pytest.raises(SinkError):
        sink.read_parquet("no-bucket-key")


# ---------------------------------------------------------------------------
# AzureBlobSink (in-memory fake BlobServiceClient, tanpa jaringan)
# ---------------------------------------------------------------------------
class _HttpError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class _FakeBlobItem:
    name: str
    metadata: dict[str, str] | None = None


class _FakeDownload:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def readall(self) -> bytes:
        return self._data


class _FakeBlobClient:
    def __init__(self, store: dict[tuple[str, str], bytes], container: str, blob: str) -> None:
        self._store = store
        self._key = (container, blob)

    def exists(self) -> bool:
        return self._key in self._store

    def upload_blob(self, data: bytes, *, overwrite: bool = True) -> None:
        if not overwrite and self._key in self._store:
            raise _HttpError(409, "BlobAlreadyExists")
        self._store[self._key] = bytes(data)

    def download_blob(self) -> _FakeDownload:
        if self._key not in self._store:
            raise _HttpError(404, "BlobNotFound")
        return _FakeDownload(self._store[self._key])

    def delete_blob(self) -> None:
        if self._key not in self._store:
            raise _HttpError(404, "BlobNotFound")
        del self._store[self._key]


class _FakeContainerClient:
    def __init__(
        self,
        store: dict[tuple[str, str], bytes],
        folders: dict[str, list[str]],
        container: str,
    ) -> None:
        self._store = store
        self._folders = folders
        self._container = container

    def list_blobs(self, *, name_starts_with: str | None = None):
        items: list[_FakeBlobItem] = []
        for folder in self._folders.get(self._container, []):
            if name_starts_with is None or folder.startswith(name_starts_with):
                items.append(_FakeBlobItem(name=folder, metadata={"hdi_isfolder": "true"}))
        for (c, blob), _ in self._store.items():
            if c == self._container and (name_starts_with is None or blob.startswith(name_starts_with)):
                items.append(_FakeBlobItem(name=blob, metadata={}))
        return items


class _FakeServiceClient:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], bytes] = {}
        self.folders: dict[str, list[str]] = {}

    def get_blob_client(self, container: str, blob: str) -> _FakeBlobClient:
        return _FakeBlobClient(self.store, container, blob)

    def get_container_client(self, container: str) -> _FakeContainerClient:
        return _FakeContainerClient(self.store, self.folders, container)


def test_azure_blob_sink_parquet_and_no_overwrite():
    service = _FakeServiceClient()
    sink = AzureBlobSink(service_client=service)
    key = "bronze/yf/prices/ticker=JKSE/ingest_date=2026-11-13/data.parquet"

    first = pd.DataFrame({"Date": [pd.Timestamp("2026-11-13")], "Close": [7300.0]})
    second = pd.DataFrame({"Date": [pd.Timestamp("2026-11-13")], "Close": [9999.0]})

    assert sink.write_parquet(first, key, overwrite=False) is True
    assert sink.exists(key)
    assert sink.write_parquet(second, key, overwrite=False) is False
    pd.testing.assert_frame_equal(sink.read_parquet(key), first)


def test_azure_blob_sink_skips_adls_hns_directory_markers():
    service = _FakeServiceClient()
    service.folders["silver"] = ["prices", "prices/ticker=JKSE"]
    sink = AzureBlobSink(service_client=service)

    sink.write_bytes(b"abc", "silver/prices/ticker=JKSE/data.parquet")
    assert sink.read_bytes("silver/prices/ticker=JKSE/data.parquet") == b"abc"
    assert sink.list_keys("silver/") == ["silver/prices/ticker=JKSE/data.parquet"]

    sink.delete("silver/prices/ticker=JKSE/data.parquet")
    assert not sink.exists("silver/prices/ticker=JKSE/data.parquet")
    sink.delete("silver/prices/ticker=JKSE/data.parquet")  # idempoten


def test_azure_blob_sink_missing_raises_sink_error():
    sink = AzureBlobSink(service_client=_FakeServiceClient())
    with pytest.raises(SinkError):
        sink.read_parquet("gold/champion/data.parquet")


def test_settings_accepts_azure_sink(monkeypatch):
    monkeypatch.setenv("SINK", "azure")
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "saihsgkel3")
    settings = Settings.from_env()
    settings.validate()
    assert settings.azure_storage_account_url == "https://saihsgkel3.blob.core.windows.net"

    monkeypatch.delenv("AZURE_STORAGE_ACCOUNT_NAME")
    monkeypatch.delenv("AZURE_STORAGE_ACCOUNT_URL", raising=False)
    bad = Settings.from_env()
    with pytest.raises(ValueError, match="AZURE_STORAGE_ACCOUNT_URL"):
        bad.validate()
    with pytest.raises(SinkError):
        get_sink(bad)
