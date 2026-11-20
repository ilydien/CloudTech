from __future__ import annotations

import pandas as pd
import pytest

from ihsg.storage import LocalSink, SinkError


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
