"""Metadata instrumen yang dipakai sebagai fitur.

``key`` dipakai di nama kolom dan nama partisi, jadi harus aman untuk S3:
tanpa ``^``, ``=``, ``/``, dan tanpa spasi. Simbol asli yfinance tetap disimpan
di ``ticker``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ihsg.config import Settings


@dataclass(frozen=True)
class Instrument:
    ticker: str          # simbol yfinance, mis. "^JKSE"
    key: str             # nama aman untuk kolom & partisi, mis. "JKSE"
    name: str
    asset_class: str     # index | commodity | fx | equity
    calendar: str        # IDX | US | HK | JP | FX


INSTRUMENTS: tuple[Instrument, ...] = (
    Instrument("^JKSE", "JKSE", "Jakarta Composite Index", "index", "IDX"),
    Instrument("CL=F", "CLF", "Crude Oil WTI Futures", "commodity", "US"),
    Instrument("GC=F", "GCF", "Gold Futures", "commodity", "US"),
    Instrument("IDR=X", "IDRX", "USD/IDR", "fx", "FX"),
    Instrument("^HSI", "HSI", "Hang Seng Index", "index", "HK"),
    Instrument("^N225", "N225", "Nikkei 225", "index", "JP"),
    Instrument("^DJI", "DJI", "Dow Jones Industrial Average", "index", "US"),
    Instrument("EEM", "EEM", "iShares MSCI Emerging Markets ETF", "equity", "US"),
)

INDEX_TICKER = "^JKSE"
INDEX_KEY = "JKSE"

_BY_TICKER = {item.ticker: item for item in INSTRUMENTS}
_BY_KEY = {item.key: item for item in INSTRUMENTS}


def get_instruments(settings: Settings) -> tuple[Instrument, ...]:
    """Instrumen aktif, mengikuti urutan TICKERS di environment.

    Instrumen yang tidak dikenal diabaikan; kalau hasilnya kosong, jatuh ke
    seluruh daftar supaya pipeline tidak mati karena typo environment.
    """
    selected = tuple(_BY_TICKER[t] for t in settings.tickers if t in _BY_TICKER)
    if not selected:
        return INSTRUMENTS
    if INDEX_TICKER not in {item.ticker for item in selected}:
        # IHSG wajib ada: target model adalah IHSG.
        selected = (INSTRUMENTS[0], *selected)
    return selected


def by_key(key: str) -> Instrument:
    return _BY_KEY[key]


def by_ticker(ticker: str) -> Instrument:
    return _BY_TICKER[ticker]


def index_instrument() -> Instrument:
    return _BY_TICKER[INDEX_TICKER]
