"""Tahapan pipeline. Setiap modul punya satu fungsi ``run`` yang idempoten."""

from ihsg.pipelines import features, ingest, predict, silver, train

__all__ = ["features", "ingest", "predict", "silver", "train"]
