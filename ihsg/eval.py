"""Metrik & uji statistik untuk evaluasi ramalan.

Semua fungsi murni (numpy/scipy), tanpa I/O - sehingga mudah diuji dan bisa
dipakai ulang oleh notebook laporan.

Referensi:
  Bollerslev (1986)              - GARCH, mean reversion varians
  Clark & West (2007)            - uji akurasi prediktif model bersarang
  Diebold & Mariano (1995)       - perbandingan akurasi
  Harvey, Leybourne & Newbold    - koreksi sampel kecil untuk DM
  Patton (2011)                  - QLIKE robust terhadap proxy berisik
  Pesaran & Timmermann (1992)    - uji arah
"""

from __future__ import annotations

import numpy as np
from scipy import stats

EPS = 1e-4  # lantai numerik untuk varians (%^2)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def qlike(actual, forecast) -> np.ndarray:
    """QLIKE: ``a/f - log(a/f) - 1``. Lebih kecil lebih baik; 0 = sempurna."""
    a = np.maximum(np.asarray(actual, dtype=float), EPS)
    f = np.maximum(np.asarray(forecast, dtype=float), EPS)
    ratio = a / f
    return ratio - np.log(ratio) - 1.0


def mse(actual, forecast) -> np.ndarray:
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    return (a - f) ** 2


def rmse(actual, forecast) -> float:
    return float(np.sqrt(np.mean(mse(actual, forecast))))


# ---------------------------------------------------------------------------
# Uji perbandingan
# ---------------------------------------------------------------------------
def dm_test(loss_model, loss_bench, h: int = 1) -> tuple[float, float]:
    """Diebold-Mariano dengan HAC Bartlett + koreksi Harvey-Leybourne-Newbold.

    Statistik negatif berarti model lebih baik (loss lebih kecil).
    """
    d = np.asarray(loss_model, dtype=float) - np.asarray(loss_bench, dtype=float)
    n = len(d)
    if n < 3:
        return float("nan"), float("nan")
    mean_d = float(d.mean())
    centred = d - mean_d
    lag = max(h - 1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    long_run = float(np.mean(centred**2))
    for k in range(1, lag + 1):
        weight = 1.0 - k / (lag + 1.0)
        long_run += 2.0 * weight * float(np.mean(centred[k:] * centred[:-k]))
    se = np.sqrt(max(long_run / n, 1e-12))
    correction = np.sqrt((n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n)
    statistic = mean_d / se * correction
    p_value = 2.0 * (1.0 - stats.t.cdf(abs(statistic), df=n - 1))
    return float(statistic), float(p_value)


def clark_west(y, forecast_model, forecast_bench, bench_mean) -> tuple[float, float]:
    """Clark-West (2007), satu sisi. Model harus bersarang di benchmark.

    Hipotesis nol: tidak ada perbaikan akurasi. p kecil = model lebih baik.
    """
    y = np.asarray(y, dtype=float)
    f = np.asarray(forecast_model, dtype=float)
    b = np.asarray(forecast_bench, dtype=float)
    m = np.asarray(bench_mean, dtype=float)

    error_model = y - f
    adjusted = (y - m) ** 2 - (error_model**2 - (m - f) ** 2)
    sigma = float(adjusted.std(ddof=1))
    mean_adjusted = float(adjusted.mean())
    if len(adjusted) < 3:
        return float("nan"), float("nan")
    if sigma <= 1e-12:
        # Tidak ada variasi loss: model identik dengan benchmark.
        if abs(mean_adjusted) <= 1e-12:
            return 0.0, 0.5
        return float("nan"), float("nan")
    statistic = mean_adjusted / (sigma / np.sqrt(len(adjusted)))
    return statistic, float(1.0 - stats.norm.cdf(statistic))


def r2_oos(y, forecast, benchmark) -> float:
    """R^2 out-of-sample terhadap benchmark (Campbell & Thompson, 2008)."""
    y = np.asarray(y, dtype=float)
    f = np.asarray(forecast, dtype=float)
    b = np.asarray(benchmark, dtype=float)
    denominator = float(np.sum((y - b) ** 2))
    if denominator <= 0:
        return float("nan")
    return 1.0 - float(np.sum((y - f) ** 2)) / denominator


def pesaran_timmermann(actual_up, predicted_up) -> tuple[float, float]:
    """Uji arah non-parametrik. NaN bila model hanya menebak satu arah."""
    y = np.asarray(actual_up, dtype=bool)
    x = np.asarray(predicted_up, dtype=bool)
    n = len(y)
    if n == 0:
        return float("nan"), float("nan")
    py, px = float(y.mean()), float(x.mean())
    if px in (0.0, 1.0) or py in (0.0, 1.0):
        return float("nan"), float("nan")
    hit_rate = float((y == x).mean())
    expected = py * px + (1.0 - py) * (1.0 - px)
    var_hit = expected * (1.0 - expected) / n
    var_expected = (
        (2.0 * py - 1.0) ** 2 * px * (1.0 - px)
        + (2.0 * px - 1.0) ** 2 * py * (1.0 - py)
        + 4.0 * py * px * (1.0 - py) * (1.0 - px) / n
    ) / n
    denominator = var_hit - var_expected
    if denominator <= 0:
        return float("nan"), float("nan")
    statistic = (hit_rate - expected) / np.sqrt(denominator)
    return float(statistic), float(1.0 - stats.norm.cdf(statistic))


# ---------------------------------------------------------------------------
# Klasifikasi rezim
# ---------------------------------------------------------------------------
def auc_rank(labels, scores) -> float:
    """AUC via statistik Mann-Whitney (tanpa sklearn)."""
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=float)
    positives = int(y.sum())
    negatives = int((~y).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # rata-rata peringkat untuk nilai kembar
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts), dtype=float)
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    rank_sum = float(ranks[y].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def regime_metrics(actual_high, prob_high, threshold: float = 0.5) -> dict[str, float]:
    """Precision/Recall/F1/AUC untuk kelas HIGH.

    Tanpa base rate, "akurasi" tinggi hanya berarti mayoritas hari NORMAL.
    """
    actual = np.asarray(actual_high, dtype=bool)
    prob = np.asarray(prob_high, dtype=float)
    predicted = prob >= threshold

    true_positive = int((predicted & actual).sum())
    precision = true_positive / int(predicted.sum()) if predicted.sum() > 0 else float("nan")
    recall = true_positive / int(actual.sum()) if actual.sum() > 0 else float("nan")
    denominator = precision + recall if np.isfinite(precision) and np.isfinite(recall) else 0.0
    f1 = 2.0 * precision * recall / denominator if denominator > 0 else float("nan")

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc_rank(actual, prob)),
        "base_rate": float(actual.mean()) if len(actual) else float("nan"),
        "accuracy": float((predicted == actual).mean()) if len(actual) else float("nan"),
    }
