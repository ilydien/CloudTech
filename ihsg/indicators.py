"""Rumus indikator yang dipakai lintas layer (silver dan gold).

Dikumpulkan di satu tempat supaya notebook laporan, silver, dan fitur tidak
punya definisi yang berbeda untuk hal yang sama.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_return_pct(price: pd.Series) -> pd.Series:
    """Log-return harian dalam persen: ``100 * ln(p_t / p_{t-1})``.

    Rasio non-positif menghasilkan NaN, bukan warning: futures seperti WTI
    pernah ditutup negatif, dan log-return di titik itu memang tidak terdefinisi.
    """
    ratio = price / price.shift(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        logged = 100.0 * np.log(ratio.where(ratio > 0))
    return logged


def parkinson_var(high: pd.Series, low: pd.Series) -> pd.Series:
    """Varians Parkinson (1980) dari High-Low, dalam satuan %^2.

    ``1e4 * ln(H/L)^2 / (4 ln 2)``. Karena memakai rasio, split saham saling
    menghapus sehingga tidak perlu penyesuaian harga.
    """
    ratio = np.maximum(high / np.maximum(low, 1e-8), 1.0)
    return 1e4 * np.log(ratio) ** 2 / (4.0 * np.log(2.0))


def sq_ret(returns_pct: pd.Series) -> pd.Series:
    return returns_pct**2


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ewma_var(returns_pct: pd.Series, decay: float) -> pd.Series:
    """EWMA RiskMetrics pada return kuadrat."""
    values = returns_pct.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) == 0:
        return pd.Series(out, index=returns_pct.index)
    seed = int(min(30, len(values)))
    out[seed - 1] = np.nanvar(values[:seed], ddof=1) if seed > 1 else values[0] ** 2
    for index in range(seed, len(values)):
        out[index] = decay * out[index - 1] + (1.0 - decay) * values[index - 1] ** 2
    return pd.Series(out, index=returns_pct.index)


def expanding_var(returns_pct: pd.Series, min_periods: int = 30) -> pd.Series:
    """Varians historis expanding, digeser satu hari (hanya informasi t-1)."""
    return returns_pct.expanding(min_periods=min_periods).var().shift(1)
