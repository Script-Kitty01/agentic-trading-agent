"""
Order Execution & Trade Lifecycle Manager.
Executes Iron Condor legs, monitors fills, manages profit target / stop loss exits,
and persists trade history into SQLite.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from alpacha.config import Settings
from alpacha.data.alpaca_data import AlpacaDataClient
from alpacha.data.sqlite_manager import SQLiteManager
from alpacha.strategy.ironcondor import IronCondor, OptionLeg
from alpacha.utils.logger import get_logger

logger = get_logger("executor")


class OrderExecutor:
    def __init__(
        self,
        settings: Settings,
        alpaca_client: AlpacaDataClient,
        db_manager: SQLiteManager,
    ) -> None:
        self.settings = settings
        self.client = alpaca_client
        self.db = db_manager
        self.dry_run = settings.app.dry_run

    def execute_iron_condor(
        self,
        condor: IronCondor,
        contracts: int,
        gates_passed: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Executes the 4 legs of an Iron Condor.
        In dry-run mode, logs and records fill immediately.
        In live/paper mode, submits limit orders and tracks status.
        """
        condor.contracts = contracts
        trade_id = condor.trade_id
        symbol = condor.underlying

        logger.info(
            f"Executing Iron Condor [{trade_id}] for {symbol}: {contracts} contracts. "
            f"Net Credit=${condor.net_credit_per_share:.2f}/share, Total=${condor.net_credit_total * contracts:.2f}"
        )

        if self.dry_run or not self.client.is_connected:
            logger.info(f"[DRY RUN] Simulating immediate fill for Iron Condor {trade_id}")
            self.db.save_trade(
                trade_id=trade_id,
                symbol=symbol,
                status="OPEN",
                entry_timestamp=datetime.now(timezone.utc),
                legs=condor.to_dict()["legs"],
                credit_received=condor.net_credit_per_share * contracts * 100.0,
                gates_passed=gates_passed,
            )
            return True, None

        # Live / Paper Execution on Alpaca
        submitted_orders = []
        try:
            for leg in [condor.long_put, condor.short_put, condor.short_call, condor.long_call]:
                side = OrderSide.BUY if leg.action == "BUY" else OrderSide.SELL
                limit_price = round(leg.mid, 2)
                order_req = LimitOrderRequest(
                    symbol=leg.symbol,
                    qty=contracts,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price,
                )
                order = self.client.trading_client.submit_order(order_data=order_req)
                submitted_orders.append(order)
                logger.info(f"Submitted order for {leg.symbol} ({leg.action} {contracts} @ ${limit_price})")

            # Poll for fills
            all_filled = self._wait_for_fills(submitted_orders, timeout_sec=self.settings.execution.order_poll_timeout_sec)
            if not all_filled:
                logger.warning(f"Orders for {trade_id} did not fill completely within timeout. Cancelling open orders.")
                for ord in submitted_orders:
                    try:
                        self.client.trading_client.cancel_order_by_id(ord.id)
                    except Exception:
                        pass
                return False, "Order fill timeout reached. Partial fill protection invoked."

            # Save filled trade in database
            self.db.save_trade(
                trade_id=trade_id,
                symbol=symbol,
                status="OPEN",
                entry_timestamp=datetime.now(timezone.utc),
                legs=condor.to_dict()["legs"],
                credit_received=condor.net_credit_per_share * contracts * 100.0,
                gates_passed=gates_passed,
            )
            return True, None

        except Exception as e:
            logger.error(f"Error executing Iron Condor {trade_id}: {e}", exc_info=True)
            return False, str(e)

    def _wait_for_fills(self, orders: List[Any], timeout_sec: int = 60) -> bool:
        """Polls submitted orders until all are filled or timeout occurs."""
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            all_done = True
            for ord in orders:
                try:
                    current_ord = self.client.trading_client.get_order_by_id(ord.id)
                    if str(current_ord.status).lower() not in ["filled"]:
                        all_done = False
                        break
                except Exception as e:
                    logger.warning(f"Error polling order {ord.id}: {e}")
                    all_done = False
                    break
            if all_done:
                return True
            time.sleep(self.settings.execution.order_poll_interval_sec)
        return False

    def check_trade_exit_condition(
        self,
        trade: Dict[str, Any],
        current_close_cost: float,
        current_dte: int,
    ) -> Tuple[bool, Optional[str], float]:
        """
        Evaluates a trade against profit targets and stop loss exit criteria.
        Supports both Iron Condor (Credit) and Momentum Breakout Call Debit Spreads.
        """
        credit_or_debit = float(trade.get("credit_received", 0.0))
        trade_id = str(trade.get("trade_id", ""))
        is_momentum = "MOM" in trade_id or credit_or_debit < 0

        if is_momentum:
            # Momentum Call Debit Spread: initial cost was debit_paid > 0
            debit_paid = abs(credit_or_debit)
            current_value = current_close_cost  # proceeds from closing spread
            pnl = current_value - debit_paid

            # Profit Target (+35%)
            target_pct = getattr(self.settings.strategy, "momentum_profit_target_pct", 0.35)
            target_gain = debit_paid * target_pct
            if pnl >= target_gain:
                return True, f"Momentum Profit Target Hit: +{pnl/debit_paid:.1%} (+${pnl:.2f}) >= +{target_pct:.0%}", pnl

            # Stop Loss (-15%)
            stop_pct = getattr(self.settings.strategy, "momentum_stop_loss_pct", 0.15)
            max_loss = debit_paid * stop_pct
            if pnl <= -max_loss:
                return True, f"Momentum Stop Loss Hit: -${-pnl:.2f} >= -${max_loss:.2f} (-{stop_pct:.0%})", pnl

            return False, None, pnl

        # Standard Iron Condor (Credit) Spread
        initial_credit = credit_or_debit
        pnl = initial_credit - current_close_cost

        # 1. Hard Gamma Exit (<= close_dte_threshold, if close_dte_threshold > 0)
        close_dte = getattr(self.settings.strategy, "close_dte_threshold", 0)
        if close_dte > 0 and current_dte <= close_dte:
            return True, f"DTE Gamma Hard Exit ({current_dte} DTE remaining <= {close_dte})", pnl

        # 2. Profit Target (e.g. 30% max profit)
        target_pnl = initial_credit * self.settings.strategy.profit_target_pct
        if pnl >= target_pnl:
            return True, f"Profit Target Hit: PnL ${pnl:.2f} >= Target ${target_pnl:.2f} ({self.settings.strategy.profit_target_pct:.0%})", pnl

        # 3. Stop Loss (e.g. 1.0x credit received loss)
        max_loss = initial_credit * self.settings.strategy.stop_loss_multiplier
        if pnl <= -max_loss:
            return True, f"Stop Loss Triggered: Loss ${-pnl:.2f} >= Max Loss ${max_loss:.2f} ({self.settings.strategy.stop_loss_multiplier:.1f}x credit)", pnl

        # 4. Management DTE Exit (if exit_dte_threshold > 0)
        exit_dte = getattr(self.settings.strategy, "exit_dte_threshold", 0)
        if exit_dte > 0 and current_dte <= exit_dte and pnl > 0:
            return True, f"Theta Management Exit: {current_dte} DTE remaining, capturing profit ${pnl:.2f}", pnl

        return False, None, pnl

    def evaluate_open_positions(
        self,
        current_time: Optional[datetime] = None,
        close_costs: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluates all open Iron Condor positions against exit criteria and closes eligible ones.
        """
        now_dt = current_time or datetime.now(timezone.utc)
        now_date = now_dt.date()
        closed_trades = []
        # Only monitor trades that match the current run mode (dry-run vs live)
        mode = "DRY_RUN" if self.dry_run else "LIVE"
        open_trades = self.db.get_open_trades(mode=mode)

        for trade in open_trades:
            trade_id = trade["trade_id"]
            symbol = trade["symbol"]
            legs = trade.get("legs", [])
            initial_credit = float(trade.get("credit_received", 0.0))

            if not legs:
                continue

            min_dte = 999
            for leg in legs:
                exp_str = leg.get("expiration")
                if exp_str:
                    try:
                        exp_d = datetime.strptime(exp_str, "%Y-%m-%d").date()
                        dte = (exp_d - now_date).days
                        if dte < min_dte:
                            min_dte = dte
                    except Exception:
                        pass

            # Cost to close if provided or default estimate
            current_close_cost = (close_costs or {}).get(trade_id, initial_credit * 0.50 if min_dte <= self.settings.strategy.close_dte_threshold else initial_credit)
            should_exit, reason, pnl = self.check_trade_exit_condition(trade, current_close_cost, min_dte)

            if should_exit:
                logger.info(f"Closing trade {trade_id} ({symbol}): Reason={reason}, Exit PnL=${pnl:.2f}")
                self.db.update_trade_exit(
                    trade_id=trade_id,
                    exit_timestamp=now_dt,
                    exit_pnl=pnl,
                    status="CLOSED",
                )
                closed_trades.append({
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "exit_reason": reason,
                    "exit_pnl": pnl,
                    "exit_timestamp": now_dt.isoformat(),
                })

        return closed_trades
