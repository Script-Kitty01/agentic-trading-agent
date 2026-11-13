import numpy as np
import pandas as pd
from alpacha.config import Settings
from alpacha.strategy.momentum import TrendMomentumEngine, MomentumSignal


def test_momentum_engine_bullish_breakout():
    settings = Settings.load(config_path="config/settings.yaml")
    engine = TrendMomentumEngine(settings)

    # Generate 30 bars with strong upward price breakout and volume surge
    dates = pd.date_range("2025-01-01 09:30:00", periods=30, freq="1min")
    prices = np.linspace(100.0, 115.0, 30)  # Steady upward trend > 20 SMA
    volumes = [500] * 25 + [3000] * 5       # Volume breakout on last 5 bars

    df = pd.DataFrame({
        "open": prices,
        "high": prices + 0.5,
        "low": prices - 0.5,
        "close": prices,
        "volume": volumes,
    }, index=dates)

    signal = engine.evaluate_momentum("NVDA", df)
    assert signal is not None
    assert signal.signal_type == "BULLISH_BREAKOUT"
    assert signal.spot_price == 115.0
    assert signal.spot_price > signal.sma_20
    assert signal.target_price > signal.spot_price
    assert signal.stop_price < signal.spot_price
    assert signal.score >= 0.50


def test_momentum_call_and_put_spreads():
    settings = Settings.load(config_path="config/settings.yaml")
    engine = TrendMomentumEngine(settings)

    mock_chain = [
        {"symbol": "NVDA260902C00115000", "option_type": "CALL", "strike": 115.0, "expiration": "2026-09-02", "dte": 2, "price": 3.50},
        {"symbol": "NVDA260902C00118000", "option_type": "CALL", "strike": 118.0, "expiration": "2026-09-02", "dte": 2, "price": 1.50},
        {"symbol": "NVDA260902P00115000", "option_type": "PUT", "strike": 115.0, "expiration": "2026-09-02", "dte": 2, "price": 3.20},
        {"symbol": "NVDA260902P00112000", "option_type": "PUT", "strike": 112.0, "expiration": "2026-09-02", "dte": 2, "price": 1.20},
    ]

    # 1. Bullish Call Debit Spread
    call_spread = engine.build_bullish_call_spread("NVDA", 115.0, mock_chain)
    assert call_spread is not None
    assert call_spread["trade_type"] == "MOMENTUM_CALL_SPREAD"
    assert call_spread["net_debit_per_share"] == 2.00  # $3.50 - $1.50
    assert len(call_spread["legs"]) == 2
    assert call_spread["legs"][0]["action"] == "BUY"
    assert call_spread["legs"][1]["action"] == "SELL"

    # 2. Bearish Put Debit Spread
    put_spread = engine.build_bearish_put_spread("NVDA", 115.0, mock_chain)
    assert put_spread is not None
    assert put_spread["trade_type"] == "MOMENTUM_PUT_SPREAD"
    assert put_spread["net_debit_per_share"] == 2.00  # $3.20 - $1.20
    assert len(put_spread["legs"]) == 2
    assert put_spread["legs"][0]["action"] == "BUY"
    assert put_spread["legs"][1]["action"] == "SELL"
