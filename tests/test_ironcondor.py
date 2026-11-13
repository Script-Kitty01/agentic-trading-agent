from alpacha.config import Settings
from alpacha.strategy.ironcondor import IronCondorBuilder


def test_iron_condor_builder():
    settings = Settings.load(config_path="config/settings.yaml")
    settings.strategy.min_dte = 1
    settings.strategy.max_dte = 45
    builder = IronCondorBuilder(settings)

    spot = 500.0
    vol = 0.20
    target_dte = 2
    exp_date = "2026-09-03"

    # Mock option chain
    chain = [
        # Puts
        {"symbol": "SPY_P_470", "option_type": "PUT", "strike": 470.0, "expiration": exp_date, "dte": 2, "delta": -0.10, "bid": 0.40, "ask": 0.50},
        {"symbol": "SPY_P_480", "option_type": "PUT", "strike": 480.0, "expiration": exp_date, "dte": 2, "delta": -0.20, "bid": 1.20, "ask": 1.30},
        {"symbol": "SPY_P_490", "option_type": "PUT", "strike": 490.0, "expiration": exp_date, "dte": 2, "delta": -0.35, "bid": 2.50, "ask": 2.60},
        # Calls
        {"symbol": "SPY_C_510", "option_type": "CALL", "strike": 510.0, "expiration": exp_date, "dte": 2, "delta": 0.35, "bid": 2.50, "ask": 2.60},
        {"symbol": "SPY_C_520", "option_type": "CALL", "strike": 520.0, "expiration": exp_date, "dte": 2, "delta": 0.20, "bid": 1.20, "ask": 1.30},
        {"symbol": "SPY_C_530", "option_type": "CALL", "strike": 530.0, "expiration": exp_date, "dte": 2, "delta": 0.10, "bid": 0.40, "ask": 0.50},
    ]

    ic = builder.build_iron_condor(
        symbol="SPY",
        underlying_price=spot,
        implied_vol=vol,
        chain_data=chain,
        dte_target=target_dte,
    )

    assert ic is not None
    assert ic.underlying == "SPY"
    assert ic.short_put.strike == 480.0
    assert ic.short_call.strike == 520.0
    assert ic.long_put.strike < 480.0
    assert ic.long_call.strike > 520.0
    assert ic.net_credit_per_share > 0.20
    assert ic.max_loss_total > 0
    assert ic.put_breakeven < 480.0
    assert ic.call_breakeven > 520.0


def test_order_executor_exit_conditions():
    from alpacha.data.alpaca_data import AlpacaDataClient
    from alpacha.data.sqlite_manager import SQLiteManager
    from alpacha.strategy.executor import OrderExecutor

    settings = Settings.load(config_path="config/settings.yaml")
    settings.strategy.close_dte_threshold = 2
    db = SQLiteManager(":memory:")
    client = AlpacaDataClient(settings)
    executor = OrderExecutor(settings, client, db)

    mock_trade = {
        "trade_id": "test_ic_001",
        "symbol": "SPY",
        "status": "OPEN",
        "credit_received": 200.0,  # $2.00/share x 100
        "legs": [
            {"symbol": "SPY260220P00480000", "expiration": "2026-02-20", "action": "SELL"},
        ],
    }

    # 1. Profit Target Check (e.g. 30% profit target: PnL = $110 >= $60)
    # Cost to close = $90 -> PnL = $200 - $90 = $110
    exit_pnl, reason, pnl = executor.check_trade_exit_condition(mock_trade, current_close_cost=90.0, current_dte=15)
    assert exit_pnl is True
    assert "Profit Target Hit" in reason
    assert pnl == 110.0

    # 2. Stop Loss Check (e.g. 1.0x loss: initial credit $200 -> max loss $200)
    # Cost to close = $650 -> PnL = $200 - $650 = -$450 <= -$200
    exit_sl, sl_reason, sl_pnl = executor.check_trade_exit_condition(mock_trade, current_close_cost=650.0, current_dte=15)
    assert exit_sl is True
    assert "Stop Loss Triggered" in sl_reason
    assert sl_pnl == -450.0

    # 3. DTE Gamma Exit Check (e.g. DTE = 2 <= close_dte_threshold 2)
    exit_dte, dte_reason, _ = executor.check_trade_exit_condition(mock_trade, current_close_cost=150.0, current_dte=2)
    assert exit_dte is True
    assert "DTE Gamma" in dte_reason
