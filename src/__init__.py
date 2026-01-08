"""
Rivalland Prop Trading System
==============================
Production-grade swing trading system based on Marc Rivalland's methodology.

This package provides:
- Core trading engine with swing point detection
- Order state machine with fill tracking
- Comprehensive risk management
- Zerodha Kite Connect integration
- Production backtesting with slippage/commissions
- Real-time monitoring and notifications
"""

__version__ = "2.0.0"
__author__ = "Prop Trading Team"

from src.config.settings import get_settings, Settings
from src.core.engine import RivallandSwingEngine
from src.core.orchestrator import TradingOrchestrator

__all__ = [
    "get_settings",
    "Settings",
    "RivallandSwingEngine",
    "TradingOrchestrator",
]
