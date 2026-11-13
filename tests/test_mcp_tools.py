import json
from alpacha.mcp.tools import (
    alpaca_get_account,
    alpaca_get_positions,
    alpaca_submit_limit_order,
    alpaca_cancel_all_orders,
    alpaca_close_all_positions,
    get_alpaca_mcp_tools,
)


def test_alpaca_mcp_tools_list():
    tools = get_alpaca_mcp_tools()
    assert len(tools) >= 5
    tool_names = [t.name for t in tools]
    assert "alpaca_get_account" in tool_names
    assert "alpaca_submit_limit_order" in tool_names
    assert "alpaca_close_all_positions" in tool_names


def test_alpaca_mcp_account_tool():
    res_str = alpaca_get_account.invoke({})
    data = json.loads(res_str)
    assert "equity" in data
    assert "buying_power" in data
    assert data["equity"] > 0


def test_alpaca_mcp_order_tool():
    res_str = alpaca_submit_limit_order.invoke({
        "symbol": "SPY",
        "qty": 1,
        "side": "buy",
        "limit_price": 500.0,
        "time_in_force": "day",
    })
    data = json.loads(res_str)
    assert "symbol" in data
    assert data["symbol"] == "SPY"


def test_alpaca_mcp_cancel_all_orders_tool():
    res_str = alpaca_cancel_all_orders.invoke({})
    data = json.loads(res_str)
    assert data["status"] == "cancelled"
    assert "result" in data


def test_cli_driver_cancel_all_orders():
    from alpacha.cli.driver import AlpacaCLIDriver
    from alpacha.config import Settings
    settings = Settings.load(config_path="config/settings.yaml")
    driver = AlpacaCLIDriver(settings)
    # Offline / mock CLI returns empty list without error
    res = driver.cancel_all_orders()
    assert isinstance(res, list)
