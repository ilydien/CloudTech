"""Fixture bersama. Tidak ada test yang menyentuh jaringan atau Docker."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    """Pastikan test tidak mewarisi environment pengembang."""
    for name in (
        "AS_OF", "SINK", "S3_ENDPOINT_URL", "S3_ACCESS_KEY", "S3_SECRET_KEY",
        "GARAGE_DEFAULT_ACCESS_KEY", "GARAGE_DEFAULT_SECRET_KEY", "IDX_HOLIDAYS",
        "TICKERS", "ADMIN_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SINK", "local")
    monkeypatch.setenv("LOCAL_ROOT", os.fspath(Path("/tmp") / "ihsg-test-root"))
    yield
