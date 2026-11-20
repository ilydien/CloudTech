"""Baseline ekonometrik untuk pembanding model deep learning.

Semua baseline memprediksi **varians hari t** memakai informasi sampai t-1,
pada skala yang sama dengan proxy (``c_train * parkinson``), sehingga
perbandingan adil tanpa kalibrasi khusus per model.

Referensi:
  Bollerslev (1986)  GARCH
  Corsi (2009)       HAR-RV
  Duan (1983)        smearing untuk transformasi balik log
  Glosten dkk (1993) GJR-GARCH
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ihsg.eval import EPS


# ---------------------------------------------------------------------------
# Baseline sederhana
# ---------------------------------------------------------------------------
def historical_var(returns_pct: np.ndarray, min_periods: int = 30) -> np.ndarray:
    series = pd.Series(returns_pct)
    return series.expanding(min_periods=min_periods).var().shift(1).to_numpy(dtype=float)


def ewma_var(returns_pct: np.ndarray, decay: float = 0.94, seed: int = 30) -> np.ndarray:
    values = np.asarray(returns_pct, dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) <= seed:
        return out
    out[seed - 1] = float(np.var(values[:seed], ddof=1))
    for index in range(seed, len(values)):
        out[index] = decay * out[index - 1] + (1.0 - decay) * values[index - 1] ** 2
    return out


# ---------------------------------------------------------------------------
# HAR-log dengan smearing Duan
# ---------------------------------------------------------------------------
def _har_design(tv: np.ndarray) -> np.ndarray:
    logged = pd.Series(np.log(np.maximum(tv, EPS)))
    lag1 = logged.shift(1)
    ma5 = lag1.rolling(5).mean()
    ma22 = lag1.rolling(22).mean()
    return np.column_stack([lag1.to_numpy(), ma5.to_numpy(), ma22.to_numpy()])


def har_log_walk_forward(
    tv: np.ndarray,
    s0: int,
    s1: int,
    refit_every: int = 21,
) -> np.ndarray:
    """Regresi log-varians pada rata-rata 1/5/22 hari, refit tiap ``refit_every``."""
    design = _har_design(tv)
    target = np.log(np.maximum(tv, EPS))
    out = np.full(s1 - s0, np.nan)

    for start in range(s0, s1, refit_every):
        end = min(start + refit_every, s1)
        rows = np.isfinite(design[:start]).all(axis=1) & np.isfinite(target[:start])
        if rows.sum() < 30:
            out[start - s0 : end - s0] = np.nanmean(tv[:start]) if start else np.nan
            continue

        matrix = np.column_stack([np.ones(int(rows.sum())), design[:start][rows]])
        coefficients = np.linalg.lstsq(matrix, target[:start][rows], rcond=None)[0]
        residuals = target[:start][rows] - matrix @ coefficients
        smearing = float(np.mean(np.exp(residuals)))

        block = design[start:end]
        if not np.isfinite(block).all():
            out[start - s0 : end - s0] = np.exp(np.log(np.maximum(tv[start - 1], EPS)))
            continue
        forecast = np.column_stack([np.ones(len(block)), block]) @ coefficients
        out[start - s0 : end - s0] = np.exp(forecast) * smearing

    return out


# ---------------------------------------------------------------------------
# GARCH / GJR-GARCH
# ---------------------------------------------------------------------------
def _fit_garch(returns_pct: np.ndarray, kind: str = "garch") -> dict:
    from arch import arch_model

    specification = arch_model(
        returns_pct,
        mean="Constant",
        vol="GARCH",
        p=1,
        o=1 if kind == "gjr" else 0,
        q=1,
        dist="t",
        rescale=False,
    )
    result = specification.fit(disp="off", update_freq=0)
    params = result.params
    alpha = float(params["alpha[1]"])
    beta = float(params["beta[1]"])
    omega = float(params["omega"])
    gamma = float(params["gamma[1]"]) if kind == "gjr" else 0.0
    persistence = alpha + 0.5 * gamma + beta
    unconditional = omega / (1.0 - persistence) if persistence < 0.999 else float(np.var(returns_pct))
    return {
        "mu": float(params["mu"]),
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "nu": float(params["nu"]),
        "persistence": float(persistence),
        "uncond": float(unconditional),
        "cond_var": np.asarray(result.conditional_volatility, dtype=float) ** 2,
    }


def _extend(fit: dict, returns_pct: np.ndarray, start: int, end: int) -> np.ndarray:
    """Rekursi varians satu langkah dengan parameter tetap dan return aktual."""
    previous = fit["cond_var"][-1]
    out = np.empty(end - start)
    for position, t in enumerate(range(start, end)):
        shock = returns_pct[t - 1] - fit["mu"]
        variance = (
            fit["omega"]
            + (fit["alpha"] + fit["gamma"] * (shock < 0)) * shock**2
            + fit["beta"] * previous
        )
        out[position] = variance
        previous = variance
    return out


def garch_walk_forward(
    returns_pct: np.ndarray,
    s0: int,
    s1: int,
    kind: str = "garch",
    refit_every: int = 21,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ramalan varians walk-forward. Return ``(variance, uncond, persistence)``."""
    out = np.full(s1 - s0, np.nan)
    uncond = np.full(s1 - s0, np.nan)
    persistence = np.full(s1 - s0, np.nan)
    previous_fit: dict | None = None

    for start in range(s0, s1, refit_every):
        end = min(start + refit_every, s1)
        try:
            fit = _fit_garch(returns_pct[:start], kind)
            previous_fit = fit
        except Exception:  # noqa: BLE001 - konvergensi GARCH bisa gagal
            if previous_fit is None:
                # Fallback: EWMA. Jangan pernah mengembalikan NaN ke evaluasi.
                out[start - s0 : end - s0] = ewma_var(returns_pct, 0.94)[start:end]
                uncond[start - s0 : end - s0] = float(np.nanvar(returns_pct[:start]))
                persistence[start - s0 : end - s0] = 0.94
                continue
            fit = previous_fit

        out[start - s0 : end - s0] = _extend(fit, returns_pct, start, end)
        uncond[start - s0 : end - s0] = fit["uncond"]
        persistence[start - s0 : end - s0] = fit["persistence"]

    return out, uncond, persistence


# ---------------------------------------------------------------------------
# Ramalan satu langkah untuk sesi berikutnya (dipakai pemilihan model)
# ---------------------------------------------------------------------------
def hist_forecast_next(returns_pct: np.ndarray, n_real: int) -> float:
    """Varians historis seluruh riwayat sampai ``n_real-1``."""
    series = np.asarray(returns_pct[:n_real], dtype=float)
    finite = series[np.isfinite(series)]
    if finite.size < 30:
        return float("nan")
    return float(np.var(finite, ddof=1))


def ewma_forecast_next(returns_pct: np.ndarray, n_real: int, decay: float = 0.94) -> float:
    series = np.asarray(returns_pct[:n_real], dtype=float)
    history = ewma_var(series, decay)
    if not np.isfinite(history[-1]):
        return hist_forecast_next(returns_pct, n_real)
    return float(decay * history[-1] + (1.0 - decay) * series[-1] ** 2)


def har_forecast_next(tv: np.ndarray, n_real: int) -> float:
    """HAR-log satu langkah: fit pada seluruh riwayat, prediksi baris ``n_real``."""
    design = _har_design(tv)
    target = np.log(np.maximum(np.asarray(tv, dtype=float), EPS))
    rows = np.isfinite(design[:n_real]).all(axis=1) & np.isfinite(target[:n_real])
    if int(rows.sum()) < 30 or not np.isfinite(design[n_real]).all():
        return float("nan")
    matrix = np.column_stack([np.ones(int(rows.sum())), design[:n_real][rows]])
    coefficients = np.linalg.lstsq(matrix, target[:n_real][rows], rcond=None)[0]
    residuals = target[:n_real][rows] - matrix @ coefficients
    smearing = float(np.mean(np.exp(residuals)))
    forecast = float(np.concatenate([[1.0], design[n_real]]) @ coefficients)
    return float(np.exp(forecast) * smearing)


def garch_forecast_next(
    returns_pct: np.ndarray, n_real: int, kind: str = "garch"
) -> tuple[float, float, float]:
    """GARCH satu langkah. Return ``(variance_h1, uncond, persistence)``."""
    series = np.asarray(returns_pct[:n_real], dtype=float)
    finite = np.isfinite(series)
    if int(finite.sum()) < 100:
        fallback = hist_forecast_next(returns_pct, n_real)
        return fallback, fallback, 0.94
    fit = _fit_garch(series[finite], kind)
    shock = float(series[n_real - 1]) - fit["mu"]
    variance = (
        fit["omega"]
        + (fit["alpha"] + fit["gamma"] * (shock < 0)) * shock**2
        + fit["beta"] * float(fit["cond_var"][-1])
    )
    return float(variance), float(fit["uncond"]), float(fit["persistence"])
