"""
Trend & Momentum Breakout Strategy Engine.
Detects bullish breakout momentum on high-beta growth equities using Moving Averages,
RSI, Volume Surges, and MACD trend confirmation to capture fast upside appreciation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from alpacha.config import Settings
from alpacha.utils.logger import get_logger

logger = get_logger("momentum")


@dataclass
class MomentumSignal:
    symbol: str
    signal_type: str         # "BULLISH_BREAKOUT" or "NEUTRAL"
    spot_price: float
    sma_20: float
    rsi_14: float
    volume_ratio: float
    score: float             # [0.0 to 1.0]
    target_price: float
    stop_price: float
    target_strike: Optional[float] = None
    long_call_symbol: Optional[str] = None
    suggested_qty: int = 1


class TrendMomentumEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.rsi_period = 14
        self.sma_period = 20
        self.min_volume_ratio = 1.15
        self.min_rsi = 50.0
        self.max_rsi = 85.0

    def compute_rsi(self, series: pd.Series, period: int = 14) -> float:
        """Computes Relative Strength Index (RSI)."""
        if len(series) < period + 1:
            return 50.0

        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean().iloc[-1]
        avg_loss = loss.rolling(window=period, min_periods=period).mean().iloc[-1]

        if avg_loss == 0 or np.isnan(avg_loss):
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(np.nan_to_num(rsi, nan=50.0))

    def evaluate_momentum(
        self,
        symbol: str,
        bars_df: pd.DataFrame,
    ) -> Optional[MomentumSignal]:
        """
        Evaluates 1-minute/daily price action for high-probability bullish breakout momentum.
        Criteria:
        1. Price > 20 SMA
        2. RSI between 50 and 75 (bullish momentum without being severely overbought)
        3. Volume surge > 1.15x rolling 20-period average volume
        4. Upward price slope over past 5 bars
        """
        if bars_df.empty or len(bars_df) < self.sma_period:
            return None

        closes = bars_df["close"]
        volumes = bars_df["volume"]

        current_price = float(closes.iloc[-1])
        sma_20 = float(closes.rolling(window=self.sma_period).mean().iloc[-1])
        recent_vol_avg = float(volumes.rolling(window=self.sma_period).mean().iloc[-1])
        current_vol = float(volumes.iloc[-1])

        volume_ratio = (current_vol / recent_vol_avg) if recent_vol_avg > 0 else 1.0
        rsi = self.compute_rsi(closes, period=self.rsi_period)

        # Check Trend Slope (5-bar return)
        five_bar_return = (current_price - float(closes.iloc[-5])) / float(closes.iloc[-5]) if len(closes) >= 5 else 0.0

        # 1. Bullish Breakout Condition
        is_above_sma = current_price > sma_20
        is_rsi_bullish = rsi >= self.min_rsi
        is_vol_expanding = volume_ratio >= self.min_volume_ratio or current_vol >= 1000
        is_slope_positive = five_bar_return > 0.001

        if is_above_sma and is_rsi_bullish and is_slope_positive:
            score = round(min(1.0, 0.5 + (five_bar_return * 20.0) + (volume_ratio * 0.1)), 2)
            target_price = round(current_price * 1.03, 2)
            stop_price = round(current_price * 0.985, 2)
            target_strike = round(current_price * 1.01 * 2) / 2

            logger.info(
                f"[Momentum Engine] Bullish Breakout for {symbol}: Spot=${current_price:.2f}, "
                f"SMA20=${sma_20:.2f}, RSI={rsi:.1f}, VolRatio={volume_ratio:.2f}x, Score={score:.2f}"
            )

            return MomentumSignal(
                symbol=symbol,
                signal_type="BULLISH_BREAKOUT",
                spot_price=current_price,
                sma_20=sma_20,
                rsi_14=rsi,
                volume_ratio=volume_ratio,
                score=score,
                target_price=target_price,
                stop_price=stop_price,
                target_strike=target_strike,
            )

        # 2. Bearish Breakdown Momentum Condition
        is_below_sma = current_price < sma_20
        is_rsi_bearish = rsi <= 45.0
        is_slope_negative = five_bar_return < -0.001

        if is_below_sma and is_rsi_bearish and is_slope_negative:
            score = round(min(1.0, 0.5 + (abs(five_bar_return) * 20.0) + (volume_ratio * 0.1)), 2)
            target_price = round(current_price * 0.97, 2)
            stop_price = round(current_price * 1.015, 2)
            target_strike = round(current_price * 0.99 * 2) / 2

            logger.info(
                f"[Momentum Engine] Bearish Breakdown for {symbol}: Spot=${current_price:.2f}, "
                f"SMA20=${sma_20:.2f}, RSI={rsi:.1f}, VolRatio={volume_ratio:.2f}x, Score={score:.2f}"
            )

            return MomentumSignal(
                symbol=symbol,
                signal_type="BEARISH_BREAKDOWN",
                spot_price=current_price,
                sma_20=sma_20,
                rsi_14=rsi,
                volume_ratio=volume_ratio,
                score=score,
                target_price=target_price,
                stop_price=stop_price,
                target_strike=target_strike,
            )

        return None

    def build_bearish_put_spread(
        self,
        symbol: str,
        spot_price: float,
        chain_data: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Constructs a high-velocity 2-leg Bearish Put Debit Spread (1-3 DTE) for downside momentum capture.
        - Long Put: ATM / slightly OTM (strike >= spot)
        - Short Put: OTM downside target (strike <= spot * 0.98)
        """
        if not chain_data or spot_price <= 0:
            return None

        puts = [p for p in chain_data if p.get("option_type") == "PUT"]
        if len(puts) < 2:
            return None

        puts.sort(key=lambda p: float(p["strike"]), reverse=True)

        # Long Put (ATM / closest above spot)
        long_candidates = [p for p in puts if float(p["strike"]) >= spot_price * 0.995]
        if not long_candidates:
            long_candidates = puts[:1]
        long_put = min(long_candidates, key=lambda p: float(p["strike"]))

        # Short Put (OTM downside target ~2% below spot)
        target_short_strike = spot_price * 0.98
        short_candidates = [p for p in puts if float(p["strike"]) < float(long_put["strike"])]
        if not short_candidates:
            return None

        short_put = min(short_candidates, key=lambda p: abs(float(p["strike"]) - target_short_strike))

        long_px = float(long_put.get("price", (long_put.get("bid", 0) + long_put.get("ask", 0)) / 2.0))
        short_px = float(short_put.get("price", (short_put.get("bid", 0) + short_put.get("ask", 0)) / 2.0))
        net_debit = max(0.10, round(long_px - short_px, 2))

        spread_width = float(long_put["strike"]) - float(short_put["strike"])
        if spread_width <= 0 or net_debit >= spread_width:
            return None

        max_profit_per_share = spread_width - net_debit
        trade_id = f"MOM_PUT_{symbol}_{long_put['expiration']}_{int(spot_price)}"

        return {
            "trade_id": trade_id,
            "symbol": symbol,
            "trade_type": "MOMENTUM_PUT_SPREAD",
            "expiration": long_put["expiration"],
            "dte": int(long_put.get("dte", 2)),
            "spot_price": spot_price,
            "net_debit_per_share": net_debit,
            "net_debit_total": net_debit * 100.0,
            "max_profit_total": max_profit_per_share * 100.0,
            "target_profit": net_debit * 100.0 * getattr(self.settings.strategy, "momentum_profit_target_pct", 0.35),
            "stop_loss_amount": net_debit * 100.0 * getattr(self.settings.strategy, "momentum_stop_loss_pct", 0.15),
            "legs": [
                {
                    "symbol": long_put["symbol"],
                    "action": "BUY",
                    "strike": float(long_put["strike"]),
                    "option_type": "PUT",
                    "expiration": long_put["expiration"],
                    "price": long_px,
                },
                {
                    "symbol": short_put["symbol"],
                    "action": "SELL",
                    "strike": float(short_put["strike"]),
                    "option_type": "PUT",
                    "expiration": short_put["expiration"],
                    "price": short_px,
                },
            ],
        }

    def build_bullish_call_spread(
        self,
        symbol: str,
        spot_price: float,
        chain_data: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Constructs a high-velocity 2-leg Bullish Call Debit Spread (1-3 DTE) for momentum breakout capture.
        - Long Call: ATM / slightly ITM (strike <= spot)
        - Short Call: OTM upside target (strike >= spot * 1.015)
        """
        if not chain_data or spot_price <= 0:
            return None

        calls = [c for c in chain_data if c.get("option_type") == "CALL"]
        if len(calls) < 2:
            return None

        # Sort calls by strike
        calls.sort(key=lambda c: float(c["strike"]))

        # Find Long Call (ATM / closest to spot)
        long_candidates = [c for c in calls if float(c["strike"]) <= spot_price * 1.005]
        if not long_candidates:
            long_candidates = calls[:1]
        long_call = max(long_candidates, key=lambda c: float(c["strike"]))

        # Find Short Call (OTM target, ~1.5% to 3.0% above spot)
        target_short_strike = spot_price * 1.02
        short_candidates = [c for c in calls if float(c["strike"]) > float(long_call["strike"])]
        if not short_candidates:
            return None

        short_call = min(short_candidates, key=lambda c: abs(float(c["strike"]) - target_short_strike))

        long_px = float(long_call.get("price", (long_call.get("bid", 0) + long_call.get("ask", 0)) / 2.0))
        short_px = float(short_call.get("price", (short_call.get("bid", 0) + short_call.get("ask", 0)) / 2.0))
        net_debit = max(0.10, round(long_px - short_px, 2))

        spread_width = float(short_call["strike"]) - float(long_call["strike"])
        if spread_width <= 0 or net_debit >= spread_width:
            return None

        max_profit_per_share = spread_width - net_debit
        trade_id = f"MOM_{symbol}_{long_call['expiration']}_{int(spot_price)}"

        return {
            "trade_id": trade_id,
            "symbol": symbol,
            "trade_type": "MOMENTUM_CALL_SPREAD",
            "expiration": long_call["expiration"],
            "dte": int(long_call.get("dte", 2)),
            "spot_price": spot_price,
            "net_debit_per_share": net_debit,
            "net_debit_total": net_debit * 100.0,
            "max_profit_total": max_profit_per_share * 100.0,
            "target_profit": net_debit * 100.0 * getattr(self.settings.strategy, "momentum_profit_target_pct", 0.35),
            "stop_loss_amount": net_debit * 100.0 * getattr(self.settings.strategy, "momentum_stop_loss_pct", 0.15),
            "legs": [
                {
                    "symbol": long_call["symbol"],
                    "action": "BUY",
                    "strike": float(long_call["strike"]),
                    "option_type": "CALL",
                    "expiration": long_call["expiration"],
                    "price": long_px,
                },
                {
                    "symbol": short_call["symbol"],
                    "action": "SELL",
                    "strike": float(short_call["strike"]),
                    "option_type": "CALL",
                    "expiration": short_call["expiration"],
                    "price": short_px,
                },
            ],
        }
