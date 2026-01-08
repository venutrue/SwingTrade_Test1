"""
Pytest Configuration and Shared Fixtures
==========================================
Shared fixtures and configuration for all tests.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def mock_database():
    """Mock database for all tests."""
    with pytest.MonkeyPatch.context() as mp:
        mock_db = MagicMock()
        mock_db.session = MagicMock()
        mp.setattr("src.core.database.db", mock_db, raising=False)
        yield mock_db


@pytest.fixture
def sample_settings():
    """Create sample settings for testing."""
    from decimal import Decimal

    settings = MagicMock()

    # Risk settings
    settings.risk = MagicMock()
    settings.risk.max_risk_per_trade_percent = Decimal("1.0")
    settings.risk.max_position_size_percent = Decimal("10.0")
    settings.risk.max_open_positions = 5
    settings.risk.min_risk_reward_ratio = Decimal("2.0")
    settings.risk.max_daily_loss_percent = Decimal("3.0")
    settings.risk.max_daily_trades = 10
    settings.risk.consecutive_loss_limit = 3
    settings.risk.max_sector_exposure_percent = Decimal("30.0")
    settings.risk.max_single_stock_percent = Decimal("15.0")
    settings.risk.brokerage_per_order = Decimal("20")
    settings.risk.stt_percent = Decimal("0.1")
    settings.risk.expected_slippage_percent = Decimal("0.1")
    settings.risk.circuit_breaker_cooldown_minutes = 30

    # Strategy settings
    settings.strategy = MagicMock()
    settings.strategy.swing_lookback = 3
    settings.strategy.min_pullback_bars = 2
    settings.strategy.min_rally_bars = 2
    settings.strategy.trailing_stop_activation_percent = Decimal("2.0")
    settings.strategy.trailing_stop_distance_percent = Decimal("1.0")

    # Market settings
    settings.market = MagicMock()
    settings.market.timezone = "Asia/Kolkata"
    settings.market.market_open_time = "09:15"
    settings.market.market_close_time = "15:30"
    settings.market.trading_start_buffer_minutes = 5
    settings.market.trading_end_buffer_minutes = 5
    settings.market.exchange = MagicMock()
    settings.market.exchange.value = "NSE"

    return settings
