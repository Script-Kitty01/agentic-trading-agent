"""
AlpachaBot Master Orchestrator.
Coordinates market data ingestion, HAR volatility forecasting, risk management ladder,
entry gate evaluations, Iron Condor generation, order execution, and position lifecycle management.
"""

from __future__ import annotations

import math
import signal
import sys
import time

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from alpacha.agent.state_machine import TradingStateMachineBuilder
from alpacha.alerts.notifier import Notifier
from alpacha.cli.driver import AlpacaCLIDriver
from alpacha.config import Settings
from alpacha.data.macro_calendar import MacroCalendar
from alpacha.data.sqlite_manager import SQLiteManager
from alpacha.model.trainer import ModelTrainer
from alpacha.risk.risk_manager import RiskLevel, RiskManager
from alpacha.strategy.gates import StrategyGateManager
from alpacha.strategy.ironcondor import IronCondorBuilder
from alpacha.utils.logger import get_logger
from alpacha.utils.time_utils import is_market_open

logger = get_logger("bot")


class AlpachaBot:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.is_running = False
        self.is_halted = False

        # Initialize subsystems
        self.db = SQLiteManager(settings.app.db_path)
        self.macro_calendar = MacroCalendar(settings.app.macro_calendar_path)
        self.cli_driver = AlpacaCLIDriver(settings)
        self.trainer = ModelTrainer(settings, self.db)
        self.risk_manager = RiskManager(settings, self.db)
        self.gate_manager = StrategyGateManager(settings, self.macro_calendar)
        self.ic_builder = IronCondorBuilder(settings)
        self.notifier = Notifier(settings)

        # Build LangGraph Agent State Machine
        self.agent_builder = TradingStateMachineBuilder(settings, self.db)
        self.agent_graph = self.agent_builder.build_graph()

        self.scheduler = BlockingScheduler()
        logger.info(f"Initialized {settings.app.name} Agent (Paper={settings.app.paper}, DryRun={settings.app.dry_run})")

    def run_cycle(self) -> None:
        """Executes a complete cycle via the LangGraph State Machine."""
        logger.info("=== Starting LangGraph Agent Execution Cycle ===")

        if self.settings.execution.trading_hours_only and not is_market_open():
            logger.info("Market is currently closed. Skipping trading scan.")
            return

        if self.is_halted:
            logger.critical("Agent is in HALTED state due to Kill Switch breach. No operations performed.")
            return

        init_state = {
            "symbols": self.settings.data.symbols,
            "step_history": [],
            "is_market_open": is_market_open(),
        }

        try:
            result = self.agent_graph.invoke(init_state)
            logger.info(f"LangGraph Agent Completed Steps: {result.get('step_history')}")
            if result.get("is_halted"):
                self.is_halted = True
                self.notifier.notify_kill_switch_triggered(result.get("drawdown_pct", 0.0), result.get("current_equity", 0.0))
            elif result.get("risk_level") == "WARNING":
                self.notifier.notify_drawdown_warning(result.get("drawdown_pct", 0.0), result.get("current_equity", 0.0))
        except Exception as e:
            logger.error(f"Error in LangGraph Agent execution: {e}", exc_info=True)

        logger.info("=== Finished LangGraph Agent Execution Cycle ===")

    def start(self) -> None:
        self.is_running = True
        logger.info(f"Starting {self.settings.app.name} Daemon...")

        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)

        cron_min = self.settings.execution.scan_cron_minutes
        self.scheduler.add_job(
            self.run_cycle,
            trigger=CronTrigger(minute=cron_min),
            id="market_scan",
            name="Market Scan & Risk Cycle",
            replace_existing=True,
        )

        try:
            self.run_cycle()
        except Exception as e:
            logger.error(f"Error in initial startup cycle: {e}", exc_info=True)

        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            self._handle_exit(None, None)

    def _handle_exit(self, signum: Any, frame: Any) -> None:
        logger.info("Received termination signal. Shutting down daemon gracefully...")
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        self.is_running = False
        sys.exit(0)

