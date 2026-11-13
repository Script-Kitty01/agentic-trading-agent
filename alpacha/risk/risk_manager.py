"""
Risk Manager for AlpachaBot.
Tracks equity high-water mark, evaluates drawdown ladder (2% warning, 3.5% kill switch),
manages position sizing, and enforces portfolio guardrails.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Tuple

from alpacha.config import Settings
from alpacha.data.sqlite_manager import SQLiteManager
from alpacha.utils.logger import get_logger

logger = get_logger("risk_manager")


class RiskLevel(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    KILL = "KILL"


@dataclass
class RiskStatus:
    current_equity: float
    peak_equity: float
    drawdown_pct: float
    risk_level: RiskLevel
    can_trade: bool
    should_liquidate: bool
    message: str


class RiskManager:
    META_PEAK_EQUITY_KEY = "peak_account_equity"

    def __init__(self, settings: Settings, db_manager: SQLiteManager) -> None:
        self.settings = settings
        self.db = db_manager
        self.warn_drawdown = settings.risk.warn_drawdown_pct
        self.kill_drawdown = settings.risk.kill_drawdown_pct
        self.max_portfolio_bp_pct = settings.risk.max_portfolio_bp_pct
        self.max_contracts = settings.risk.max_contracts_per_trade

        # Load or initialize peak equity
        self._peak_equity = self._load_peak_equity()

    def _load_peak_equity(self) -> float:
        val = self.db.get_meta(self.META_PEAK_EQUITY_KEY)
        if val:
            try:
                return float(val)
            except ValueError:
                pass
        return 0.0

    def update_peak_equity(self, current_equity: float) -> float:
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
            self.db.set_meta(self.META_PEAK_EQUITY_KEY, str(current_equity))
            logger.info(f"New Equity High-Water Mark: ${self._peak_equity:,.2f}")
        return self._peak_equity

    def evaluate_risk(self, current_equity: float) -> RiskStatus:
        """
        Evaluates portfolio drawdown against risk ladder:
        - Drawdown < 2.0%: NORMAL
        - Drawdown >= 2.0% and < 3.5%: WARNING (block new entries, warn)
        - Drawdown >= 3.5%: KILL (liquidate all positions, halt daemon)
        """
        if current_equity <= 0.0:
            logger.critical("Current equity is 0 or negative!")
            return RiskStatus(
                current_equity=current_equity,
                peak_equity=self._peak_equity,
                drawdown_pct=1.0,
                risk_level=RiskLevel.KILL,
                can_trade=False,
                should_liquidate=True,
                message="Account equity is 0 or negative.",
            )

        # Update peak if equity grows
        peak = self.update_peak_equity(current_equity)

        # Calculate drawdown from peak
        drawdown_pct = max(0.0, (peak - current_equity) / peak) if peak > 0 else 0.0

        if drawdown_pct >= self.kill_drawdown:
            level = RiskLevel.KILL
            can_trade = False
            should_liquidate = True
            msg = f"CRITICAL: Drawdown {drawdown_pct:.2%} exceeds kill threshold {self.kill_drawdown:.2%}. Triggering liquidation."
            logger.critical(msg)
        elif drawdown_pct >= self.warn_drawdown:
            level = RiskLevel.WARNING
            can_trade = False
            should_liquidate = False
            msg = f"WARNING: Drawdown {drawdown_pct:.2%} exceeds warning threshold {self.warn_drawdown:.2%}. New entries halted."
            logger.warning(msg)
        else:
            level = RiskLevel.NORMAL
            can_trade = True
            should_liquidate = False
            msg = f"Risk normal: Equity=${current_equity:,.2f}, Peak=${peak:,.2f}, Drawdown={drawdown_pct:.2%}"
            logger.debug(msg)

        # Record snapshot in SQLite
        self.db.save_risk_snapshot(
            equity=current_equity,
            peak_equity=peak,
            drawdown_pct=drawdown_pct,
            risk_level=level.value,
        )

        return RiskStatus(
            current_equity=current_equity,
            peak_equity=peak,
            drawdown_pct=drawdown_pct,
            risk_level=level,
            can_trade=can_trade,
            should_liquidate=should_liquidate,
            message=msg,
        )

    @staticmethod
    def get_asset_class(symbol: str) -> str:
        """Classifies symbol into an asset category."""
        sym = symbol.upper()
        if sym in ["TLT", "IEF", "SHY", "BND", "AGG"]:
            return "FIXED_INCOME"
        elif sym in ["GLD", "SLV", "USO", "UNG", "XLE", "DBA", "DBC"]:
            return "COMMODITIES"
        return "EQUITIES"

    def check_asset_caps(
        self,
        symbol: str,
        proposed_risk: float,
        current_equity: float,
        open_positions: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Enforces AssetClassCapsConfig exposure limits against open positions and proposed risk.
        """
        if current_equity <= 0:
            return False, "Account equity is 0 or negative."

        caps = self.settings.risk.caps
        asset_class = self.get_asset_class(symbol)

        if open_positions is None:
            open_positions = self.db.get_open_trades(mode="LIVE") if self.db is not None else []

        symbol_exposure = 0.0
        class_exposure = 0.0

        for pos in open_positions:
            pos_sym = (pos.get("symbol") or "").upper()
            pos_risk = float(pos.get("credit_received", 0.0) or pos.get("risk", 0.0) or pos.get("market_value", 0.0) or 0.0)
            if pos_sym == symbol.upper():
                symbol_exposure += pos_risk
            if self.get_asset_class(pos_sym) == asset_class:
                class_exposure += pos_risk

        # 1. Single Symbol Cap Check
        max_symbol_risk = current_equity * caps.single_symbol_max_pct
        if (symbol_exposure + proposed_risk) > max_symbol_risk:
            return False, (
                f"Single symbol exposure cap exceeded for {symbol}: "
                f"Proposed total ${(symbol_exposure + proposed_risk):,.2f} > Max allowed ${max_symbol_risk:,.2f} "
                f"({caps.single_symbol_max_pct:.1%} of equity)"
            )

        # 2. Asset Class Cap Check
        if asset_class == "EQUITIES":
            class_cap_pct = caps.equity_max_pct
        elif asset_class == "FIXED_INCOME":
            class_cap_pct = caps.fixed_income_max_pct
        elif asset_class == "COMMODITIES":
            class_cap_pct = caps.commodities_max_pct
        else:
            class_cap_pct = caps.equity_max_pct

        max_class_risk = current_equity * class_cap_pct
        if (class_exposure + proposed_risk) > max_class_risk:
            return False, (
                f"Asset class exposure cap exceeded for {asset_class} ({symbol}): "
                f"Proposed total ${(class_exposure + proposed_risk):,.2f} > Max allowed ${max_class_risk:,.2f} "
                f"({class_cap_pct:.1%} of equity)"
            )

        return True, None

    def calculate_position_size(
        self,
        account_equity: float,
        available_bp: float,
        wing_width: float,
        credit_per_share: float,
        symbol: Optional[str] = None,
        forecasted_vol: Optional[float] = None,
        open_positions: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[int, Optional[str]]:
        """
        Calculates safe number of Iron Condor contracts to trade.
        Applies Risk Parity / Inverse Volatility weighting when use_risk_parity is enabled.
        Max loss per contract = (Wing Width - Credit) * 100
        Returns contracts = 0 when risk budget < 1 contract max loss.
        """
        if account_equity <= 0 or wing_width <= 0:
            return 0, "Invalid equity or wing width."

        max_loss_per_share = max(0.01, wing_width - credit_per_share)
        max_loss_per_contract = max_loss_per_share * 100.0

        # Base Risk Budget (e.g. 3% of account equity)
        risk_budget = account_equity * self.settings.risk.single_trade_max_loss_pct

        # Apply Risk Parity / Volatility Weighting
        if self.settings.risk.use_risk_parity and forecasted_vol and forecasted_vol > 0.01:
            target_vol = self.settings.risk.target_annualized_vol  # e.g. 0.15 (15%)
            vol_multiplier = max(0.25, min(2.5, target_vol / forecasted_vol))
            risk_budget *= vol_multiplier
            logger.info(
                f"Risk Parity Sizing for {symbol or 'Asset'}: Forecasted Vol={forecasted_vol:.2%}, "
                f"Target Vol={target_vol:.2%}, Vol Multiplier={vol_multiplier:.2f}x, Adjusted Budget=${risk_budget:,.2f}"
            )

        # Check single trade risk budget
        size_by_risk = int(risk_budget / max_loss_per_contract)
        if size_by_risk < 1:
            return 0, f"Risk budget (${risk_budget:.2f}) is less than max loss per contract (${max_loss_per_contract:.2f})"

        # Max buying power allocation (e.g. 30% of available buying power)
        bp_budget = available_bp * self.max_portfolio_bp_pct
        size_by_bp = int(bp_budget / max_loss_per_contract)
        if size_by_bp < 1:
            return 0, f"Buying power budget (${bp_budget:.2f}) is less than max loss per contract (${max_loss_per_contract:.2f})"

        # Cap by configured maximum contracts per trade
        contracts = min(size_by_risk, size_by_bp, self.max_contracts)
        if contracts <= 0:
            return 0, "Calculated contracts is 0."

        # Verify against asset class and single symbol exposure caps
        if symbol:
            total_risk = contracts * max_loss_per_contract
            caps_ok, cap_err = self.check_asset_caps(
                symbol=symbol,
                proposed_risk=total_risk,
                current_equity=account_equity,
                open_positions=open_positions,
            )
            if not caps_ok:
                while contracts > 1:
                    contracts -= 1
                    total_risk = contracts * max_loss_per_contract
                    caps_ok, _ = self.check_asset_caps(
                        symbol=symbol,
                        proposed_risk=total_risk,
                        current_equity=account_equity,
                        open_positions=open_positions,
                    )
                    if caps_ok:
                        break
                if not caps_ok:
                    return 0, cap_err

        total_risk = contracts * max_loss_per_contract
        if total_risk > available_bp:
            return 0, f"Insufficient buying power: requires ${total_risk:,.2f}, available ${available_bp:,.2f}"

        logger.info(
            f"Sized Iron Condor ({symbol or 'Trade'}): {contracts} contracts "
            f"(Max loss/contract=${max_loss_per_contract:.2f}, Total risk=${total_risk:.2f})"
        )
        return contracts, None

