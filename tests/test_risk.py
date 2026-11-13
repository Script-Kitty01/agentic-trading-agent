from alpacha.config import Settings
from alpacha.data.sqlite_manager import SQLiteManager
from alpacha.risk.risk_manager import RiskLevel, RiskManager


def test_risk_manager_drawdown_ladder():
    settings = Settings.load(config_path="config/settings.yaml")
    db = SQLiteManager(":memory:")
    rm = RiskManager(settings, db)

    # 1. Normal State ($100,000)
    s1 = rm.evaluate_risk(100000.0)
    assert s1.risk_level == RiskLevel.NORMAL
    assert s1.can_trade is True
    assert s1.should_liquidate is False

    # 2. Equity High Water Mark increases to $110,000
    s2 = rm.evaluate_risk(110000.0)
    assert s2.peak_equity == 110000.0

    # 3. 2.2% Drawdown from peak ($110,000 -> $107,500) -> WARNING
    s3 = rm.evaluate_risk(107500.0)
    assert s3.risk_level == RiskLevel.WARNING
    assert s3.can_trade is False
    assert s3.should_liquidate is False

    # 4. 3.8% Drawdown from peak ($110,000 -> $105,800) -> KILL SWITCH
    s4 = rm.evaluate_risk(105800.0)
    assert s4.risk_level == RiskLevel.KILL
    assert s4.can_trade is False
    assert s4.should_liquidate is True


def test_position_sizing():
    settings = Settings.load(config_path="config/settings.yaml")
    db = SQLiteManager(":memory:")
    rm = RiskManager(settings, db)

    # Account with $100k equity, $50k BP, wing width 10, credit 1.50
    # Max loss per contract = (10 - 1.5) * 100 = $850
    contracts, err = rm.calculate_position_size(
        account_equity=100000.0,
        available_bp=50000.0,
        wing_width=10.0,
        credit_per_share=1.50,
    )
    assert err is None
    assert 1 <= contracts <= settings.risk.max_contracts_per_trade


def test_strict_risk_budget_sizing_zero_contract():
    settings = Settings.load(config_path="config/settings.yaml")
    settings.risk.single_trade_max_loss_pct = 0.01  # 1% risk budget
    db = SQLiteManager(":memory:")
    rm = RiskManager(settings, db)

    # Small account: $5,000 equity -> 1% budget is $50
    # Wing width 10, credit 1.00 -> Max loss per contract is $900
    # Budget $50 < $900 -> contracts must be 0
    contracts, err = rm.calculate_position_size(
        account_equity=5000.0,
        available_bp=20000.0,
        wing_width=10.0,
        credit_per_share=1.00,
    )
    assert contracts == 0
    assert err is not None
    assert "less than max loss per contract" in err


def test_asset_class_caps_enforcement():
    settings = Settings.load(config_path="config/settings.yaml")
    settings.risk.caps.single_symbol_max_pct = 0.08  # 8% max per ticker ($8,000 on $100k equity)
    settings.risk.caps.equity_max_pct = 0.50         # 50% max equities ($50,000 on $100k equity)
    db = SQLiteManager(":memory:")
    rm = RiskManager(settings, db)

    # 1. Test single symbol cap breach
    # Equity $100k -> 8% max = $8,000. Proposed risk = $9,000 -> Should fail
    ok, err = rm.check_asset_caps(
        symbol="SPY",
        proposed_risk=9000.0,
        current_equity=100000.0,
        open_positions=[],
    )
    assert not ok
    assert "Single symbol exposure cap exceeded" in err

    # 2. Test within single symbol cap
    ok, err = rm.check_asset_caps(
        symbol="SPY",
        proposed_risk=5000.0,
        current_equity=100000.0,
        open_positions=[],
    )
    assert ok
    assert err is None

    # 3. Test asset class cap breach with existing positions
    existing_positions = [
        {"symbol": "AAPL", "credit_received": 25000.0},
        {"symbol": "MSFT", "credit_received": 22000.0},
    ]  # Total equities exposure = $47,000
    # Adding $5,000 SPY risk would total $52,000 > $50,000 (50% of $100k)
    ok, err = rm.check_asset_caps(
        symbol="SPY",
        proposed_risk=5000.0,
        current_equity=100000.0,
        open_positions=existing_positions,
    )
    assert not ok
    assert "Asset class exposure cap exceeded" in err
