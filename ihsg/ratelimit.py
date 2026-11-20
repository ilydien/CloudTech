"""Throttle & retry untuk sumber data eksternal.

yfinance memakai endpoint tidak resmi Yahoo: bisa membalas kosong, membalas
HTML error, atau membatasi laju. Modul ini menyediakan dua hal kecil yang
membuat ingestion jauh lebih tahan banting:

* ``Throttle`` - jarak minimum antar request (setara kuota per jam).
* ``retry_call`` - backoff eksponensial dengan jitter untuk kegagalan sementara.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

DEFAULT_TIMEOUT = 60.0
DEFAULT_ATTEMPTS = 4
DEFAULT_BASE_DELAY = 2.0


class Throttle:
    """Pastikan tidak lebih dari ``requests_per_hour`` permintaan per jam."""

    def __init__(self, requests_per_hour: int) -> None:
        if requests_per_hour <= 0:
            raise ValueError("requests_per_hour harus positif")
        self.min_interval = 3600.0 / requests_per_hour
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self.min_interval - (now - self._last)
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


YFINANCE_THROTTLE = Throttle(600)  # 10 request/menit, jauh di bawah batas Yahoo


def retry_call(
    fn: Callable[[], T],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = 60.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> T:
    """Jalankan ``fn`` dengan backoff eksponensial + jitter."""
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except exceptions as exc:  # noqa: BLE001 - sengaja luas, provider tidak konsisten
            last_error = exc
            if attempt >= attempts:
                break
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            delay *= 1.0 + random.uniform(0.0, 0.25)
            if on_retry is not None:
                on_retry(attempt, exc)
            time.sleep(delay)
    raise RuntimeError(f"gagal setelah {attempts} percobaan: {last_error}") from last_error
