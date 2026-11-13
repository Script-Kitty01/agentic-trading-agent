"""
LangGraph State Machine for Alpacha Options Trading Agent.
Coordinates risk evaluation, volatility forecasting, entry gating,
Iron Condor construction, and order execution via an Agent Graph.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import pandas as pd
from langgraph.graph import StateGraph, START, END

from alpacha.agent.state import TradingAgentState
from alpacha.cli.driver import AlpacaCLIDriver
from alpacha.config import Settings
from alpacha.data.macro_calendar import MacroCalendar
from alpacha.data.news_client import AlpacaNewsClient
from alpacha.data.sqlite_manager import SQLiteManager
from alpacha.model.news_analyzer import NewsSentimentAnalyzer
from alpacha.model.trainer import ModelTrainer
from alpacha.risk.risk_manager import RiskLevel, RiskManager
from alpacha.strategy.gates import StrategyGateManager
from alpacha.strategy.ironcondor import IronCondorBuilder
from alpacha.strategy.momentum import TrendMomentumEngine, MomentumSignal
from alpacha.utils.logger import get_logger

logger = get_logger("langgraph_agent")


class TradingStateMachineBuilder:
    def __init__(self, settings: Settings, db_manager: Optional[SQLiteManager] = None) -> None:
        self.settings = settings
        self.db = db_manager or SQLiteManager(settings.app.db_path)
        self.cli_driver = AlpacaCLIDriver(settings)
        self.news_client = AlpacaNewsClient(settings)
        self.news_analyzer = NewsSentimentAnalyzer()
        self.macro_calendar = MacroCalendar(settings.app.macro_calendar_path)
        self.trainer = ModelTrainer(settings, self.db)
        self.risk_manager = RiskManager(settings, self.db)
        self.gate_manager = StrategyGateManager(settings, self.macro_calendar)
        self.ic_builder = IronCondorBuilder(settings)
        self.momentum_engine = TrendMomentumEngine(settings)


    def risk_evaluation_node(self, state: TradingAgentState) -> Dict[str, Any]:
        """Node 1: Evaluates account equity against the drawdown ladder."""
        logger.info("[State Machine: Node 1] Evaluating Portfolio Risk...")
        acc = self.cli_driver.get_account()
        current_equity = float(state.get("current_equity") if state.get("current_equity") is not None else acc.get("equity", 100000.0))
        buying_power = float(state.get("buying_power") if state.get("buying_power") is not None else acc.get("buying_power", 400000.0))

        risk_status = self.risk_manager.evaluate_risk(current_equity)
        history = list(state.get("step_history", [])) + ["risk_evaluation"]

        return {
            "current_equity": current_equity,
            "peak_equity": risk_status.peak_equity,
            "buying_power": buying_power,
            "drawdown_pct": risk_status.drawdown_pct,
            "risk_level": risk_status.risk_level.value,
            "should_liquidate": risk_status.should_liquidate,
            "can_trade": risk_status.can_trade,
            "status_message": risk_status.message,
            "step_history": history,
        }

    def kill_switch_liquidation_node(self, state: TradingAgentState) -> Dict[str, Any]:
        """Node: Liquidates all positions upon 3.5% drawdown breach using Multi-Leg Limit Orders."""
        logger.critical("[State Machine: Emergency Node] Kill switch triggered! Liquidating positions via MLEG Limit Orders.")
        self.cli_driver.close_all_positions(cancel_orders=True, db_manager=self.db)
        history = list(state.get("step_history", [])) + ["kill_switch_liquidation"]
        return {
            "is_halted": True,
            "status_message": "CRITICAL KILL SWITCH EXECUTED: All positions liquidated safely via limit orders.",
            "step_history": history,
        }

    def market_data_node(self, state: TradingAgentState) -> Dict[str, Any]:
        """Node 2: Ingests 1-minute bars for target symbols."""
        logger.info("[State Machine: Node 2] Ingesting Market Data...")
        symbols = state.get("symbols", self.settings.data.symbols)
        end_ts = datetime.now(timezone.utc)
        start_ts = end_ts - timedelta(days=self.settings.data.history_days)

        bars_data = {}
        for sym in symbols:
            df = self.cli_driver.get_stock_bars(sym, start=start_ts, end=end_ts)
            if df.empty:
                df = self.db.load_bars(sym, start_ts=start_ts)
                if df.empty:
                    df = self.db.load_bars(sym)
            if not df.empty:
                self.db.save_bars(df, sym)
                bars_data[sym] = df

        history = list(state.get("step_history", [])) + ["market_data"]
        return {"bars_data": bars_data, "step_history": history}

    def momentum_scan_node(self, state: TradingAgentState) -> Dict[str, Any]:
        """Node: Scans high-beta growth assets for Bullish Momentum Breakouts."""
        logger.info("[State Machine: Node 2.5] Evaluating Trend & Momentum Breakouts...")
        bars_data = state.get("bars_data", {})
        momentum_signals = {}

        for sym, df in bars_data.items():
            try:
                sig = self.momentum_engine.evaluate_momentum(sym, df)
                if sig is not None:
                    momentum_signals[sym] = sig.__dict__
            except Exception as e:
                logger.error(f"Error scanning momentum for {sym}: {e}")

        history = list(state.get("step_history", [])) + ["momentum_scan"]
        return {"momentum_signals": momentum_signals, "step_history": history}

    def news_analysis_node(self, state: TradingAgentState) -> Dict[str, Any]:
        """Node 3: Ingests breaking news & analyzes sentiment/risk."""
        logger.info("[State Machine: Node 3] Ingesting & Analyzing Live Market News...")
        symbols = state.get("symbols", self.settings.data.symbols)
        if "news_analysis" in state and state["news_analysis"] is not None:
            news_analysis = state["news_analysis"]
        else:
            news_analysis = {}
            try:
                articles = self.news_client.get_latest_news(symbols=symbols[:10], limit=30)
                for sym in symbols:
                    sym_articles = [a for a in articles if sym in a.get("symbols", [])]
                    res = self.news_analyzer.analyze_news(sym, sym_articles)
                    news_analysis[sym] = res.__dict__
            except Exception as e:
                logger.error(f"Error in news analysis: {e}")

        history = list(state.get("step_history", [])) + ["news_analysis"]
        return {"news_analysis": news_analysis, "step_history": history}

    def volatility_forecasting_node(self, state: TradingAgentState) -> Dict[str, Any]:
        """Node 4: Fits Enhanced HAR model and generates 1-day RV forecasts."""
        logger.info("[State Machine: Node 4] Generating HAR Volatility Forecasts...")
        bars_data = state.get("bars_data", {})
        forecasts = {}

        for sym, df in bars_data.items():
            try:
                _, ann_vol = self.trainer.get_forecast(sym, df)
                forecasts[sym] = ann_vol
            except Exception as e:
                logger.error(f"Error forecasting for {sym}: {e}")

        history = list(state.get("step_history", [])) + ["volatility_forecasting"]
        return {"forecasts": forecasts, "step_history": history}

    def _extract_market_ivs(self, symbol: str, ann_vol: float) -> Tuple[float, float]:
        """Extracts real market IV for near-term and next expirations if option chain is available."""
        if not self.cli_driver.is_connected:
            return ann_vol * 1.25, ann_vol * 1.27

        try:
            today_str = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")
            data = self.cli_driver._request(
                "GET",
                "v2/options/contracts",
                params={"underlying_symbols": symbol, "expiration_date_gte": today_str, "limit": 200},
            )
            contracts = data.get("option_contracts", [])
            if contracts:
                exps = sorted(list(set(c["expiration_date"] for c in contracts if c.get("expiration_date"))))
                if len(exps) >= 2:
                    near_exp, next_exp = exps[0], exps[1]
                    near_c = [c for c in contracts if c.get("expiration_date") == near_exp and c.get("implied_volatility")]
                    next_c = [c for c in contracts if c.get("expiration_date") == next_exp and c.get("implied_volatility")]
                    if near_c and next_c:
                        n_iv = float(near_c[0]["implied_volatility"])
                        nxt_iv = float(next_c[0]["implied_volatility"])
                        if n_iv > 0.01 and nxt_iv > 0.01:
                            return n_iv, nxt_iv
        except Exception as e:
            logger.debug(f"Could not extract live IV for {symbol}: {e}")

        return ann_vol * 1.25, ann_vol * 1.27

    def gate_filtering_node(self, state: TradingAgentState) -> Dict[str, Any]:
        """Node 5: Evaluates Macro, News Sentiment, Contango, and IV/RV Edge gates."""
        logger.info("[State Machine: Node 5] Evaluating Entry & News Gates...")
        forecasts = state.get("forecasts", {})
        news_analysis = state.get("news_analysis", {})
        gate_results = {}
        eligible_symbols = []

        for sym, ann_vol in forecasts.items():
            # Check News Risk Gate
            sym_news = news_analysis.get(sym, {})
            if sym_news.get("is_event_risk_high", False):
                logger.warning(f"News Gate Blocked for {sym}: High event shock risk detected!")
                continue

            near_iv, next_iv = self._extract_market_ivs(sym, ann_vol)
            gate_res = self.gate_manager.evaluate_all_gates(
                symbol=sym,
                market_iv=near_iv,
                forecasted_rv=ann_vol,
                near_atm_iv=near_iv,
                next_atm_iv=next_iv,
                current_time=datetime.now(timezone.utc),
            )
            gate_results[sym] = gate_res.__dict__
            if gate_res.all_passed:
                eligible_symbols.append(sym)

        history = list(state.get("step_history", [])) + ["gate_filtering"]
        return {
            "gate_results": gate_results,
            "eligible_symbols": eligible_symbols,
            "step_history": history,
        }


    def iron_condor_builder_node(self, state: TradingAgentState) -> Dict[str, Any]:
        """Node 5: Builds 0.20 delta Iron Condor structures with Expected Move wings."""
        logger.info("[State Machine: Node 5] Constructing Iron Condors...")
        eligible_symbols = state.get("eligible_symbols", [])
        bars_data = state.get("bars_data", {})
        forecasts = state.get("forecasts", {})
        built_condors = {}
        sized_contracts = {}

        current_equity = state.get("current_equity", 100000.0)
        available_bp = state.get("buying_power", 400000.0)

        # Enforce portfolio capacity limits and prevent duplicate positions
        # Only LIVE trades count toward capacity caps; dry-run ghosts are ignored.
        open_trades = self.db.get_open_trades(mode="LIVE")
        max_positions = getattr(self.settings.risk, "max_open_positions", 5)
        if len(open_trades) >= max_positions:
            logger.info(f"Portfolio at maximum open position capacity ({len(open_trades)}/{max_positions}). Skipping new entries.")
            history = list(state.get("step_history", [])) + ["iron_condor_builder"]
            return {"built_condors": {}, "sized_contracts": {}, "step_history": history}

        existing_symbols = {t.get("symbol", "").upper() for t in open_trades}

        for sym in eligible_symbols:
            if len(open_trades) + len(built_condors) >= max_positions:
                logger.info(f"Reached portfolio position cap ({max_positions}). Skipping remaining symbols.")
                break

            if sym.upper() in existing_symbols:
                logger.info(f"Skipping {sym}: Already has an active open position.")
                continue

            df = bars_data.get(sym)
            if df is None or df.empty:
                continue
            spot = float(df["close"].iloc[-1])
            vol = forecasts.get(sym, 0.20)
            near_iv = vol * 1.25

            chain_list = self._format_chain(sym, spot, near_iv)
            condor = self.ic_builder.build_iron_condor(
                symbol=sym,
                underlying_price=spot,
                implied_vol=near_iv,
                chain_data=chain_list,
                dte_target=self.settings.strategy.target_dte,
            )

            if condor is not None:
                contracts, err = self.risk_manager.calculate_position_size(
                    account_equity=current_equity,
                    available_bp=available_bp,
                    wing_width=max(condor.put_wing_width, condor.call_wing_width),
                    credit_per_share=condor.net_credit_per_share,
                    symbol=sym,
                    forecasted_vol=vol,
                )
                if contracts > 0 and not err:
                    built_condors[sym] = condor.to_dict()
                    sized_contracts[sym] = contracts

        history = list(state.get("step_history", [])) + ["iron_condor_builder"]
        return {
            "built_condors": built_condors,
            "sized_contracts": sized_contracts,
            "step_history": history,
        }

    def order_executor_node(self, state: TradingAgentState) -> Dict[str, Any]:
        """Node 6: Executes limit orders via CLI/MCP Driver and records to SQLite."""
        logger.info("[State Machine: Node 6] Executing Trade Orders...")
        built_condors = state.get("built_condors", {})
        sized_contracts = state.get("sized_contracts", {})
        gate_results = state.get("gate_results", {})
        momentum_signals = state.get("momentum_signals", {})
        bars_data = state.get("bars_data", {})

        executed_trades = []
        errors = []

        for sym, condor_dict in built_condors.items():
            contracts = max(1, sized_contracts.get(sym, 1))
            trade_id = condor_dict["trade_id"]
            order_id = f"mock_{trade_id[:12]}"

            try:
                if not self.settings.app.dry_run:
                    # Construct native multi-leg (mleg) Iron Condor order for Alpaca Level 3 Options
                    mleg_legs = [
                        {
                            "symbol": leg["symbol"],
                            "ratio_qty": "1",
                            "side": "buy" if leg["action"] == "BUY" else "sell",
                        }
                        for leg in condor_dict["legs"]
                    ]
                    payload = {
                        "order_class": "mleg",
                        "qty": str(contracts),
                        "type": "limit",
                        "time_in_force": "day",
                        "limit_price": str(max(0.05, round(condor_dict.get("net_credit_per_share", 0.50), 2))),
                        "legs": mleg_legs,
                    }
                    resp = self.cli_driver._request("POST", "v2/orders", data=payload)
                    order_id = resp.get("id") if isinstance(resp, dict) else f"mleg_{trade_id[:12]}"
                    logger.info(f"Successfully submitted Multi-Leg Iron Condor {trade_id} (Alpaca Order ID: {order_id})")
                else:
                    logger.info(f"[DRY RUN] Simulating fill for trade {trade_id} ({sym} x {contracts})")

                self.db.save_trade(
                    trade_id=trade_id,
                    symbol=sym,
                    status="OPEN",
                    entry_timestamp=datetime.now(timezone.utc),
                    legs=condor_dict["legs"],
                    credit_received=condor_dict["net_credit_total"] * contracts,
                    gates_passed=gate_results.get(sym),
                    mode="DRY_RUN" if self.settings.app.dry_run else "LIVE",
                )
                executed_trades.append({
                    "trade_id": trade_id,
                    "symbol": sym,
                    "strategy": "IRON_CONDOR",
                    "contracts": contracts,
                    "order_id": order_id,
                })
            except Exception as e:
                logger.error(f"Execution error for {trade_id}: {e}")
                errors.append(str(e))

        # 2. Execute Bullish Momentum Call Debit Spreads for high-growth breakout tickers
        open_live_trades = self.db.get_open_trades(mode="LIVE" if not self.settings.app.dry_run else "DRY_RUN")
        existing_symbols = {t.get("symbol", "").upper() for t in open_live_trades}

        for sym, sig in momentum_signals.items():
            if sym.upper() in existing_symbols or sym in built_condors:
                continue

            df = bars_data.get(sym)
            if df is None or df.empty:
                continue

            spot = float(df["close"].iloc[-1])
            vol = state.get("forecasts", {}).get(sym, 0.25)
            chain = self._format_chain(sym, spot, vol)
            if not chain:
                continue

            sig_type = sig.get("signal_type", "BULLISH_BREAKOUT")
            if sig_type == "BEARISH_BREAKDOWN":
                spread = self.momentum_engine.build_bearish_put_spread(sym, spot, chain)
            else:
                spread = self.momentum_engine.build_bullish_call_spread(sym, spot, chain)

            if spread is None:
                continue

            trade_id = spread["trade_id"]
            strat_name = spread.get("trade_type", "MOMENTUM_SPREAD")
            order_id = f"mock_{trade_id[:12]}"
            qty = 1

            try:
                if not self.settings.app.dry_run:
                    mleg_legs = [
                        {"symbol": spread["legs"][0]["symbol"], "ratio_qty": "1", "side": "buy"},
                        {"symbol": spread["legs"][1]["symbol"], "ratio_qty": "1", "side": "sell"},
                    ]
                    payload = {
                        "order_class": "mleg",
                        "qty": str(qty),
                        "type": "limit",
                        "time_in_force": "day",
                        "limit_price": str(round(spread["net_debit_per_share"], 2)),
                        "legs": mleg_legs,
                    }
                    resp = self.cli_driver._request("POST", "v2/orders", data=payload)
                    order_id = resp.get("id") if isinstance(resp, dict) else f"mleg_{trade_id[:12]}"
                    logger.info(f"Successfully submitted Momentum {strat_name} {trade_id} (Alpaca Order ID: {order_id})")
                else:
                    logger.info(f"[DRY RUN] Simulating fill for momentum spread {trade_id} ({sym})")

                self.db.save_trade(
                    trade_id=trade_id,
                    symbol=sym,
                    status="OPEN",
                    entry_timestamp=datetime.now(timezone.utc),
                    legs=spread["legs"],
                    credit_received=-spread["net_debit_total"] * qty,
                    gates_passed={"strategy": strat_name, "score": sig.get("score")},
                    mode="DRY_RUN" if self.settings.app.dry_run else "LIVE",
                )
                executed_trades.append({
                    "trade_id": trade_id,
                    "symbol": sym,
                    "strategy": strat_name,
                    "contracts": qty,
                    "order_id": order_id,
                })
            except Exception as e:
                logger.error(f"Momentum execution error for {trade_id}: {e}")
                errors.append(str(e))

        history = list(state.get("step_history", [])) + ["order_executor"]
        return {
            "executed_trades": executed_trades,
            "execution_errors": errors,
            "step_history": history,
        }

    def _format_chain(self, symbol: str, spot: float, vol: float) -> List[Dict[str, Any]]:
        from scipy.stats import norm
        import math

        chain_list = []
        vol_clean = max(0.08, vol)
        today = datetime.now(timezone.utc).date()
        root = symbol.replace(".", "").upper()

        # Attempt to query live option contracts from Alpaca
        live_contracts = []
        if self.cli_driver.is_connected:
            try:
                today_str = today.strftime("%Y-%m-%d")
                calls_res = self.cli_driver._request(
                    "GET",
                    "v2/options/contracts",
                    params={
                        "underlying_symbols": symbol,
                        "type": "call",
                        "expiration_date_gte": today_str,
                        "limit": 500,
                    },
                )
                puts_res = self.cli_driver._request(
                    "GET",
                    "v2/options/contracts",
                    params={
                        "underlying_symbols": symbol,
                        "type": "put",
                        "expiration_date_gte": today_str,
                        "limit": 500,
                    },
                )
                live_contracts = calls_res.get("option_contracts", []) + puts_res.get("option_contracts", [])
            except Exception as e:
                logger.warning(f"Could not query live option contracts for {symbol}: {e}")

        if live_contracts:
            expirations = sorted(list(set(c["expiration_date"] for c in live_contracts if c.get("expiration_date"))))
            min_dte = getattr(self.settings.strategy, "min_dte", 15)
            max_dte = getattr(self.settings.strategy, "max_dte", 45)
            target_dte = getattr(self.settings.strategy, "target_dte", 30)

            valid_exps = []
            for exp in expirations:
                try:
                    exp_date_obj = datetime.fromisoformat(exp).date()
                    dte_val = (exp_date_obj - today).days
                    if min_dte <= dte_val <= max_dte:
                        valid_exps.append((exp, dte_val))
                except Exception:
                    continue

            if valid_exps:
                best_exp, actual_dte = min(valid_exps, key=lambda x: abs(x[1] - target_dte))
                t_years = max(0.01, actual_dte / 365.0)

                exp_dt = datetime.fromisoformat(best_exp).date()
                yy = exp_dt.strftime("%y")
                mm = exp_dt.strftime("%m")
                dd = exp_dt.strftime("%d")

                exp_contracts = [c for c in live_contracts if c.get("expiration_date") == best_exp]
                for c in exp_contracts:
                    raw_strike = float(c.get("strike_price", 0))
                    if raw_strike <= 0:
                        continue
                    strike = round(raw_strike * 2) / 2
                    opt_type = c.get("type", "call").upper()

                    d1 = (math.log(spot / strike) + 0.5 * (vol_clean ** 2) * t_years) / (vol_clean * math.sqrt(t_years))
                    d2 = d1 - vol_clean * math.sqrt(t_years)

                    call_delta = float(norm.cdf(d1))
                    put_delta = float(call_delta - 1.0)
                    delta = call_delta if opt_type == "CALL" else put_delta

                    if opt_type == "CALL":
                        price = max(0.10, round(spot * norm.cdf(d1) - strike * math.exp(-0.04 * t_years) * norm.cdf(d2), 2))
                    else:
                        price = max(0.10, round(strike * math.exp(-0.04 * t_years) * norm.cdf(-d2) - spot * norm.cdf(-d1), 2))

                    strike_int = int(round(strike * 1000))
                    occ_sym = c.get("symbol") or f"{root}{yy}{mm}{dd}{'C' if opt_type == 'CALL' else 'P'}{strike_int:08d}"

                    chain_list.append({
                        "symbol": occ_sym,
                        "option_type": opt_type,
                        "strike": strike,
                        "expiration": best_exp,
                        "dte": actual_dte,
                        "delta": delta,
                        "bid": max(0.05, round(price - 0.05, 2)),
                        "ask": round(price + 0.05, 2),
                        "price": price,
                        "implied_volatility": vol_clean,
                    })

                has_calls_above = any(c["option_type"] == "CALL" and c["strike"] > spot for c in chain_list)
                has_puts_below = any(c["option_type"] == "PUT" and c["strike"] < spot for c in chain_list)
                if has_calls_above and has_puts_below and len(chain_list) >= 8:
                    return chain_list

        # LIVE mode: NEVER fabricate synthetic option contracts for execution.
        # A fabricated OCC symbol (e.g. QQQ261001P00664000) does not exist on
        # Alpaca and will be rejected with a 422 "asset not found" error.
        if self.cli_driver.is_connected and not self.settings.app.dry_run:
            logger.warning(f"No tradable live option chain available for {symbol}. Skipping this symbol.")
            return []

        # Offline / synthetic testing fallback (dry-run only, never for live fills).
        target_dte = max(1, self.settings.strategy.target_dte)
        t_years = max(0.01, target_dte / 365.0)
        exp_dt = today + timedelta(days=target_dte)
        exp_date = exp_dt.strftime("%Y-%m-%d")
        yy = exp_dt.strftime("%y")
        mm = exp_dt.strftime("%m")
        dd = exp_dt.strftime("%d")
        chain_list = []

        step = 1.0 if spot >= 100.0 else 0.50
        min_strike = max(step, round((spot * 0.85) / step) * step)
        max_strike = round((spot * 1.15) / step) * step
        num_steps = max(1, int(round((max_strike - min_strike) / step)))

        for i in range(num_steps + 1):
            strike = round(min_strike + i * step, 2)
            if strike <= 0:
                continue

            d1 = (math.log(spot / strike) + 0.5 * (vol_clean ** 2) * t_years) / (vol_clean * math.sqrt(t_years))
            d2 = d1 - vol_clean * math.sqrt(t_years)

            call_delta = float(norm.cdf(d1))
            put_delta = float(call_delta - 1.0)
            call_price = max(0.10, round(spot * norm.cdf(d1) - strike * math.exp(-0.04 * t_years) * norm.cdf(d2), 2))
            put_price = max(0.10, round(strike * math.exp(-0.04 * t_years) * norm.cdf(-d2) - spot * norm.cdf(-d1), 2))

            strike_int = int(round(strike * 1000))
            chain_list.append({
                "symbol": f"{root}{yy}{mm}{dd}P{strike_int:08d}",
                "option_type": "PUT",
                "strike": strike,
                "expiration": exp_date,
                "dte": target_dte,
                "delta": put_delta,
                "bid": max(0.05, round(put_price - 0.05, 2)),
                "ask": round(put_price + 0.05, 2),
                "price": put_price,
                "implied_volatility": vol_clean,
            })
            chain_list.append({
                "symbol": f"{root}{yy}{mm}{dd}C{strike_int:08d}",
                "option_type": "CALL",
                "strike": strike,
                "expiration": exp_date,
                "dte": target_dte,
                "delta": call_delta,
                "bid": max(0.05, round(call_price - 0.05, 2)),
                "ask": round(call_price + 0.05, 2),
                "price": call_price,
                "implied_volatility": vol_clean,
            })

        return chain_list


    def build_graph(self):
        """Constructs and compiles the LangGraph State Machine with News & Sentiment Nodes."""
        graph = StateGraph(TradingAgentState)

        graph.add_node("risk_node", self.risk_evaluation_node)
        graph.add_node("kill_node", self.kill_switch_liquidation_node)
        graph.add_node("market_data_node", self.market_data_node)
        graph.add_node("momentum_node", self.momentum_scan_node)
        graph.add_node("news_node", self.news_analysis_node)
        graph.add_node("volatility_node", self.volatility_forecasting_node)
        graph.add_node("gates_node", self.gate_filtering_node)
        graph.add_node("ic_builder_node", self.iron_condor_builder_node)
        graph.add_node("executor_node", self.order_executor_node)

        def route_risk(state: TradingAgentState) -> str:
            level = state.get("risk_level", "NORMAL")
            if level == "KILL" or state.get("should_liquidate"):
                return "kill_node"
            if level == "WARNING" or not state.get("can_trade"):
                return END
            return "market_data_node"

        def route_gates(state: TradingAgentState) -> str:
            eligible = state.get("eligible_symbols", [])
            if eligible:
                return "ic_builder_node"
            return END

        graph.add_edge(START, "risk_node")
        graph.add_conditional_edges("risk_node", route_risk, {
            "kill_node": "kill_node",
            "market_data_node": "market_data_node",
            END: END,
        })
        graph.add_edge("kill_node", END)
        graph.add_edge("market_data_node", "momentum_node")
        graph.add_edge("momentum_node", "news_node")
        graph.add_edge("news_node", "volatility_node")
        graph.add_edge("volatility_node", "gates_node")
        graph.add_conditional_edges("gates_node", route_gates, {
            "ic_builder_node": "ic_builder_node",
            END: END,
        })
        graph.add_edge("ic_builder_node", "executor_node")
        graph.add_edge("executor_node", END)

        return graph.compile()


