from __future__ import annotations

import pandas as pd
import pytest

from ihsg.schemas import (
    BRONZE_PRICES,
    GOLD_PREDICTIONS,
    ContractError,
    fingerprint,
    validate_contract,
)


def _bronze_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "Date": pd.to_datetime(["2026-11-12", "2026-11-13"]),
        "Open": [100.0, 101.0],
        "High": [102.0, 103.0],
        "Low": [99.0, 100.0],
        "Close": [101.0, 102.0],
        "Adj Close": [101.0, 102.0],
        "Volume": [1000.0, 1200.0],
    })


def test_valid_bronze_contract_passes():
    validate_contract(_bronze_frame(), BRONZE_PRICES)


def test_missing_column_is_rejected():
    frame = _bronze_frame().drop(columns=["Adj Close"])
    with pytest.raises(ContractError, match="kolom hilang"):
        validate_contract(frame, BRONZE_PRICES)


def test_wrong_dtype_is_rejected():
    frame = _bronze_frame()
    frame["Close"] = frame["Close"].astype(str)
    with pytest.raises(ContractError, match="bertipe"):
        validate_contract(frame, BRONZE_PRICES)


def test_duplicate_natural_key_is_rejected():
    frame = _bronze_frame()
    frame.loc[1, "Date"] = frame.loc[0, "Date"]
    with pytest.raises(ContractError, match="duplikat"):
        validate_contract(frame, BRONZE_PRICES)


def test_null_in_required_column_is_rejected():
    frame = _bronze_frame()
    frame.loc[0, "Close"] = None
    with pytest.raises(ContractError, match="null"):
        validate_contract(frame, BRONZE_PRICES)


def test_predictions_contract_requires_horizon_keys():
    frame = pd.DataFrame({
        "as_of": ["2026-11-13", "2026-11-13"],
        "run_id": ["r1", "r1"],
        "model_vol": ["gru", "gru"],
        "model_ret": ["ridge", "ridge"],
        "base_date": ["2026-11-13", "2026-11-13"],
        "last_close": [5000.0, 5000.0],
        "horizon_step": [1, 1],
        "target_date": ["2026-11-16", "2026-11-16"],
        "forecast_variance_pct2": [4.0, 4.1],
        "forecast_volatility_pct": [2.0, 2.02],
        "cumulative_variance_pct2": [4.0, 8.1],
        "annualized_volatility_pct": [31.7, 32.1],
        "risk_regime": ["HIGH", "HIGH"],
        "prob_high": [0.6, 0.58],
        "tau": [1.81, 1.81],
        "persistence": [0.96, 0.96],
        "ml_return_enabled": [False, False],
        "expected_log_return_pct": [0.1, 0.1],
        "cumulative_expected_return_pct": [0.1, 0.2],
        "median_close_price": [5005.0, 5010.0],
        "price_interval_low": [4800.0, 4810.0],
        "price_interval_high": [5200.0, 5210.0],
        "prob_up_vs_last_close": [0.55, 0.56],
        "direction_signal": ["UP", "NO_SIGNAL"],
    })
    with pytest.raises(ContractError, match="duplikat"):
        validate_contract(frame, GOLD_PREDICTIONS)


def test_fingerprint_is_stable_and_order_sensitive():
    frame = _bronze_frame()
    assert fingerprint(frame) == fingerprint(frame.copy())
    shuffled = frame.iloc[::-1].reset_index(drop=True)
    assert fingerprint(frame) != fingerprint(shuffled)
