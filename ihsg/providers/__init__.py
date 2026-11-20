"""Sumber data eksternal."""

from ihsg.providers.yfinance_provider import RawBatch, download_raw, provider_version

__all__ = ["RawBatch", "download_raw", "provider_version"]
