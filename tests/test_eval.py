from __future__ import annotations

import numpy as np
import pytest

from ihsg import eval as metrics


def test_qlike_is_zero_when_forecast_is_perfect():
    actual = np.array([1.0, 4.0, 9.0])
    assert np.allclose(metrics.qlike(actual, actual), 0.0, atol=1e-10)


def test_qlike_penalises_underprediction_more_than_overprediction():
    actual = np.array([4.0])
    under = float(metrics.qlike(actual, np.array([1.0]))[0])
    over = float(metrics.qlike(actual, np.array([16.0]))[0])
    assert under > over


def test_dm_test_identical_losses_gives_zero_statistic():
    loss = np.linspace(0.1, 1.0, 60)
    statistic, p_value = metrics.dm_test(loss, loss)
    assert abs(statistic) < 1e-8
    assert p_value == pytest.approx(1.0, abs=1e-9)


def test_dm_test_detects_better_model():
    rng = np.random.default_rng(42)
    better = rng.uniform(0.1, 0.2, 400)
    worse = better + 0.1
    statistic, p_value = metrics.dm_test(better, worse)
    assert statistic < 0
    assert p_value < 0.01


def test_clark_west_zero_when_model_equals_benchmark():
    y = np.array([0.5, -0.2, 0.1, 0.3, -0.4, 0.2])
    bench = np.zeros_like(y)
    statistic, p_value = metrics.clark_west(y, bench, bench, bench)
    assert abs(statistic) < 1e-9
    assert p_value == pytest.approx(0.5)


def test_r2_oos_is_one_for_perfect_forecast():
    y = np.array([1.0, -1.0, 0.5])
    bench = np.zeros_like(y)
    assert metrics.r2_oos(y, y, bench) == pytest.approx(1.0)


def test_r2_oos_zero_when_model_equals_benchmark():
    y = np.array([1.0, -1.0, 0.5])
    bench = np.zeros_like(y)
    assert metrics.r2_oos(y, bench, bench) == pytest.approx(0.0)


def test_auc_rank_perfect_separation():
    labels = np.array([False, False, True, True])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert metrics.auc_rank(labels, scores) == pytest.approx(1.0)


def test_auc_rank_with_ties_is_half():
    labels = np.array([True, False])
    scores = np.array([0.5, 0.5])
    assert metrics.auc_rank(labels, scores) == pytest.approx(0.5)


def test_regime_metrics_reports_base_rate():
    actual = np.array([True, False, False, False])
    prob = np.array([0.9, 0.1, 0.2, 0.3])
    result = metrics.regime_metrics(actual, prob)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["base_rate"] == pytest.approx(0.25)


def test_pesaran_timmermann_nan_when_single_direction():
    y = np.array([True, True, True])
    x = np.array([True, True, False])
    statistic, p_value = metrics.pesaran_timmermann(y, x)
    assert np.isnan(statistic) and np.isnan(p_value)
