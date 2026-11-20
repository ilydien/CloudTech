"""Provider yfinance: unduh OHLCV **mentah** untuk bronze.

Aturan penting:
* ``auto_adjust=False`` - harga TIDAK disesuaikan di sini. Penyesuaian dan
  pemakaian ``Adj Close`` adalah urusan silver. Ini yang membuat klaim
  "bronze tidak diapa-apakan" bisa dipertahankan.
* ``actions=True`` - dividen & split ikut diambil agar penyesuaian bisa
  diaudit, bukan ditelan diam-diam.
* Nama kolom TIDAK diubah di sini (``Open``, ``High``, ...), karena bronze
  harus mencerminkan apa yang dikirim provider. Rename terjadi di silver.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from ihsg.config import JAKARTA_TZ
from ihsg.instruments import Instrument
from ihsg.ratelimit import YFINANCE_THROTTLE, retry_call

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
ACTION_COLUMNS = ["Dividends", "Stock Splits"]


@dataclass
class RawBatch:
    instrument: Instrument
    prices: pd.DataFrame
    actions: pd.DataFrame
    request_start: str
    request_end: str


def provider_version() -> str:
    try:
        import yfinance as yf

        return str(getattr(yf, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        return "not-installed"


def _split_columns(raw: pd.DataFrame) -> bool:
    return isinstance(raw.columns, pd.MultiIndex)


def _extract(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Ambil blok satu ticker dari hasil ``yf.download``."""
    if _split_columns(raw):
        if ticker in raw.columns.get_level_values(0):
            frame = raw[ticker].copy()
        elif ticker in raw.columns.get_level_values(1):
            frame = raw.xs(ticker, axis=1, level=1).copy()
        else:
            raise KeyError(f"{ticker} tidak ada di hasil unduhan")
    else:
        frame = raw.copy()
    frame.index.name = "Date"
    return frame.reset_index()


def _normalise_prices(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[["Date", *[c for c in PRICE_COLUMNS if c in frame.columns]]].copy()
    if "Adj Close" not in frame.columns:
        # Sebagian rilis yfinance tidak lagi mengirim Adj Close saat auto_adjust=False.
        frame["Adj Close"] = frame["Close"]
    if "Volume" not in frame.columns:
        frame["Volume"] = 0.0
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.tz_localize(None)
    for column in PRICE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    frame = frame.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # `yf.download` dengan banyak ticker mengembalikan index GABUNGAN semua
    # kalender, sehingga ticker AS punya baris dengan Close=NaN pada hari libur
    # BEI. Itu bukan observasi, hanya artefak batch. Buang baris tanpa Close
    # supaya isi partisi setara dengan respons per-ticker dari provider.
    return frame.dropna(subset=["Close"]).reset_index(drop=True)


def _normalise_actions(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[["Date", *[c for c in ACTION_COLUMNS if c in frame.columns]]].copy()
    for column in ACTION_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.tz_localize(None)
    for column in ACTION_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype("float64")
    return frame.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def download_raw(instruments: tuple[Instrument, ...], start: str, as_of: str) -> list[RawBatch]:
    """Unduh rentang ``start`` s/d ``as_of`` (inklusif) untuk semua instrumen.

    ``end`` pada yfinance bersifat eksklusif, jadi ditambah satu hari.
    """
    import yfinance as yf

    tickers = [item.ticker for item in instruments]
    end = (date.fromisoformat(as_of) + timedelta(days=1)).isoformat()

    def _do() -> pd.DataFrame:
        YFINANCE_THROTTLE.wait()
        raw = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            auto_adjust=False,
            actions=True,
            group_by="column",
            progress=False,
            threads=False,
            timeout=60,
        )
        if raw is None or len(raw) == 0:
            raise RuntimeError("yfinance mengembalikan data kosong")
        return raw

    raw = retry_call(_do, attempts=4, base_delay=3.0)
    if raw is None or len(raw) == 0:
        # Rentang tanpa sesi bursa (mis. hanya akhir pekan) atau provider
        # belum menerbitkan data. Bukan kegagalan; pemanggil yang memutuskan.
        return []

    batches: list[RawBatch] = []
    for item in instruments:
        try:
            frame = _extract(raw, item.ticker)
        except KeyError:
            # Satu ticker gagal tidak boleh membatalkan seluruh batch.
            continue
        prices = _normalise_prices(frame)
        actions = _normalise_actions(frame)
        if prices.empty:
            continue
        batches.append(
            RawBatch(
                instrument=item,
                prices=prices,
                actions=actions,
                request_start=start,
                request_end=end,
            )
        )
    if not batches:
        raise RuntimeError("tidak ada instrumen yang berhasil diunduh")
    return batches


def downloaded_at_utc() -> str:
    """Stempel waktu informasional (tidak dipakai untuk perbandingan hash)."""
    return datetime.now(ZoneInfo(JAKARTA_TZ)).isoformat(timespec="seconds")
