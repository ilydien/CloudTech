"""IHSG volatility & return forecasting engine.

Struktur paket:

    ihsg.config        konfigurasi dari environment (tanpa file config)
    ihsg.instruments   metadata ticker
    ihsg.dataset       konvensi path medallion + kalender bursa
    ihsg.storage       Sink (local / S3-compatible) untuk Parquet & artefak
    ihsg.schemas       kontrak kolom tiap tabel
    ihsg.quality       hard/soft data-quality + quarantine
    ihsg.eval          QLIKE, Diebold-Mariano, Clark-West, Pesaran-Timmermann
    ihsg.providers     sumber data (yfinance)
    ihsg.models        baseline ekonometrik + GRU (torch)
    ihsg.pipelines     ingest -> silver -> features -> train -> predict
    ihsg.serving       handler murni (dipakai FastAPI & Azure Function)
    ihsg.dashboard     Streamlit
"""

__version__ = "4.2.0"
