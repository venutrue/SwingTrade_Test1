"""
Unit Tests for Market Hours and Trading Calendar
=================================================
Tests for market calendar, trading hours, and holiday management.
"""

import pytest
from datetime import datetime, date, time, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.market_hours import (
    MarketCalendar, MarketSessionScheduler,
    NSE_HOLIDAYS_2025, NSE_HOLIDAYS_2026
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock()
    settings.market = MagicMock()
    settings.market.timezone = "Asia/Kolkata"
    settings.market.market_open_time = "09:15"
    settings.market.market_close_time = "15:30"
    settings.market.trading_start_buffer_minutes = 5
    settings.market.trading_end_buffer_minutes = 5
    settings.market.exchange = MagicMock()
    settings.market.exchange.value = "NSE"
    return settings


@pytest.fixture
def market_calendar(mock_settings):
    """Create a MarketCalendar instance."""
    with patch('src.utils.market_hours.get_settings', return_value=mock_settings):
        return MarketCalendar()


# ─────────────────────────────────────────────────────────────────────────────
# TEST NSE HOLIDAYS DATA
# ─────────────────────────────────────────────────────────────────────────────

class TestNSEHolidays:
    """Tests for NSE holidays data."""

    def test_2025_holidays_not_empty(self):
        """Test 2025 holidays list is not empty."""
        assert len(NSE_HOLIDAYS_2025) > 0

    def test_2026_holidays_not_empty(self):
        """Test 2026 holidays list is not empty."""
        assert len(NSE_HOLIDAYS_2026) > 0

    def test_holidays_have_valid_dates(self):
        """Test all holiday dates are valid."""
        for date_str, name in NSE_HOLIDAYS_2025 + NSE_HOLIDAYS_2026:
            # Should not raise
            parsed = datetime.strptime(date_str, "%Y-%m-%d")
            assert parsed is not None

    def test_holidays_have_names(self):
        """Test all holidays have names."""
        for date_str, name in NSE_HOLIDAYS_2025 + NSE_HOLIDAYS_2026:
            assert name is not None
            assert len(name) > 0

    def test_republic_day_exists_2025(self):
        """Test Republic Day is in 2025 holidays."""
        dates = [d for d, n in NSE_HOLIDAYS_2025 if "Republic Day" in n]
        assert len(dates) >= 1

    def test_independence_day_exists_2025(self):
        """Test Independence Day is in 2025 holidays."""
        dates = [d for d, n in NSE_HOLIDAYS_2025 if "Independence Day" in n]
        assert len(dates) >= 1

    def test_diwali_exists_2025(self):
        """Test Diwali is in 2025 holidays."""
        dates = [d for d, n in NSE_HOLIDAYS_2025 if "Diwali" in n]
        assert len(dates) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# TEST MARKET CALENDAR
# ─────────────────────────────────────────────────────────────────────────────

class TestMarketCalendar:
    """Tests for MarketCalendar class."""

    def test_is_weekend_saturday(self, market_calendar):
        """Test Saturday is detected as weekend."""
        saturday = date(2024, 1, 6)  # A Saturday
        assert market_calendar.is_weekend(saturday) is True

    def test_is_weekend_sunday(self, market_calendar):
        """Test Sunday is detected as weekend."""
        sunday = date(2024, 1, 7)  # A Sunday
        assert market_calendar.is_weekend(sunday) is True

    def test_is_weekend_monday_false(self, market_calendar):
        """Test Monday is not a weekend."""
        monday = date(2024, 1, 8)  # A Monday
        assert market_calendar.is_weekend(monday) is False

    def test_is_weekend_friday_false(self, market_calendar):
        """Test Friday is not a weekend."""
        friday = date(2024, 1, 5)  # A Friday
        assert market_calendar.is_weekend(friday) is False

    def test_get_market_open_time(self, market_calendar):
        """Test market open time is correct."""
        open_time = market_calendar.get_market_open_time()
        assert open_time == time(9, 15, 0)

    def test_get_market_close_time(self, market_calendar):
        """Test market close time is correct."""
        close_time = market_calendar.get_market_close_time()
        assert close_time == time(15, 30, 0)

    def test_get_trading_start_time_includes_buffer(self, market_calendar):
        """Test trading start time includes buffer."""
        trading_start = market_calendar.get_trading_start_time()
        # 9:15 + 5 minutes buffer = 9:20
        assert trading_start == time(9, 20, 0)

    def test_get_trading_end_time_includes_buffer(self, market_calendar):
        """Test trading end time includes buffer."""
        trading_end = market_calendar.get_trading_end_time()
        # 15:30 - 5 minutes buffer = 15:25
        assert trading_end == time(15, 25, 0)

    def test_is_market_open_during_hours(self, market_calendar):
        """Test market is open during trading hours."""
        # Mock now() to return a time during market hours
        with patch.object(market_calendar, 'now') as mock_now:
            mock_now.return_value = datetime(2024, 1, 8, 12, 0, 0)
            assert market_calendar.is_market_open() is True

    def test_is_market_open_before_open(self, market_calendar):
        """Test market is closed before open time."""
        with patch.object(market_calendar, 'now') as mock_now:
            mock_now.return_value = datetime(2024, 1, 8, 8, 0, 0)
            assert market_calendar.is_market_open() is False

    def test_is_market_open_after_close(self, market_calendar):
        """Test market is closed after close time."""
        with patch.object(market_calendar, 'now') as mock_now:
            mock_now.return_value = datetime(2024, 1, 8, 17, 0, 0)
            assert market_calendar.is_market_open() is False

    def test_is_trading_hours_during_window(self, market_calendar):
        """Test trading hours detection during window."""
        with patch.object(market_calendar, 'now') as mock_now:
            mock_now.return_value = datetime(2024, 1, 8, 12, 0, 0)
            assert market_calendar.is_trading_hours() is True

    def test_is_trading_hours_in_buffer_before(self, market_calendar):
        """Test trading hours false during start buffer."""
        with patch.object(market_calendar, 'now') as mock_now:
            # 9:17 is after market open (9:15) but before trading start (9:20)
            mock_now.return_value = datetime(2024, 1, 8, 9, 17, 0)
            assert market_calendar.is_trading_hours() is False

    def test_is_trading_hours_in_buffer_after(self, market_calendar):
        """Test trading hours false during end buffer."""
        with patch.object(market_calendar, 'now') as mock_now:
            # 15:27 is after trading end (15:25) but before market close (15:30)
            mock_now.return_value = datetime(2024, 1, 8, 15, 27, 0)
            assert market_calendar.is_trading_hours() is False

    def test_time_until_market_open_before_open(self, market_calendar):
        """Test time until market open calculation."""
        with patch.object(market_calendar, 'now') as mock_now:
            # 9:00, 15 minutes before open
            mock_now.return_value = datetime(2024, 1, 8, 9, 0, 0,
                                            tzinfo=market_calendar.timezone)
            time_until = market_calendar.time_until_market_open()

            assert time_until is not None
            assert time_until.total_seconds() == 15 * 60  # 15 minutes

    def test_time_until_market_open_after_open(self, market_calendar):
        """Test time until market open returns None after open."""
        with patch.object(market_calendar, 'now') as mock_now:
            mock_now.return_value = datetime(2024, 1, 8, 10, 0, 0,
                                            tzinfo=market_calendar.timezone)
            time_until = market_calendar.time_until_market_open()

            assert time_until is None

    def test_time_until_market_close_during_hours(self, market_calendar):
        """Test time until market close calculation."""
        with patch.object(market_calendar, 'now') as mock_now:
            # 14:30, 1 hour before close
            mock_now.return_value = datetime(2024, 1, 8, 14, 30, 0,
                                            tzinfo=market_calendar.timezone)
            time_until = market_calendar.time_until_market_close()

            assert time_until is not None
            assert time_until.total_seconds() == 60 * 60  # 1 hour

    def test_get_session_info_returns_dict(self, market_calendar):
        """Test get_session_info returns proper dictionary."""
        with patch.object(market_calendar, 'now') as mock_now:
            mock_now.return_value = datetime(2024, 1, 8, 12, 0, 0,
                                            tzinfo=market_calendar.timezone)
            info = market_calendar.get_session_info()

            assert isinstance(info, dict)
            assert "current_time" in info
            assert "timezone" in info
            assert "market_open" in info
            assert "market_close" in info
            assert "is_market_open" in info


# ─────────────────────────────────────────────────────────────────────────────
# TEST TRADING DAY CHECKS
# ─────────────────────────────────────────────────────────────────────────────

class TestTradingDayChecks:
    """Tests for trading day validation."""

    @pytest.mark.asyncio
    async def test_is_trading_day_weekend_false(self, market_calendar):
        """Test weekend is not a trading day."""
        saturday = date(2024, 1, 6)
        assert await market_calendar.is_trading_day(saturday) is False

    @pytest.mark.asyncio
    async def test_is_trading_day_weekday_true(self, market_calendar):
        """Test regular weekday is a trading day."""
        # Mock is_holiday to return False
        with patch.object(market_calendar, 'is_holiday', new_callable=AsyncMock) as mock_holiday:
            mock_holiday.return_value = False
            monday = date(2024, 1, 8)
            assert await market_calendar.is_trading_day(monday) is True

    @pytest.mark.asyncio
    async def test_can_trade_now_weekend(self, market_calendar):
        """Test can_trade_now returns False on weekend."""
        with patch.object(market_calendar, 'is_trading_day', new_callable=AsyncMock) as mock_day:
            mock_day.return_value = False
            with patch.object(market_calendar, 'is_weekend', return_value=True):
                can_trade, reason = await market_calendar.can_trade_now()
                assert can_trade is False
                assert "Weekend" in reason

    @pytest.mark.asyncio
    async def test_can_trade_now_before_market_open(self, market_calendar):
        """Test can_trade_now returns False before market open."""
        with patch.object(market_calendar, 'is_trading_day', new_callable=AsyncMock) as mock_day:
            mock_day.return_value = True
            with patch.object(market_calendar, 'is_market_open', return_value=False):
                with patch.object(market_calendar, 'now') as mock_now:
                    mock_now.return_value = datetime(2024, 1, 8, 8, 0, 0,
                                                    tzinfo=market_calendar.timezone)
                    can_trade, reason = await market_calendar.can_trade_now()
                    assert can_trade is False
                    assert "Pre-market" in reason

    @pytest.mark.asyncio
    async def test_can_trade_now_during_trading_hours(self, market_calendar):
        """Test can_trade_now returns True during trading hours."""
        with patch.object(market_calendar, 'is_trading_day', new_callable=AsyncMock) as mock_day:
            mock_day.return_value = True
            with patch.object(market_calendar, 'is_market_open', return_value=True):
                with patch.object(market_calendar, 'is_trading_hours', return_value=True):
                    can_trade, reason = await market_calendar.can_trade_now()
                    assert can_trade is True
                    assert "Trading allowed" in reason

    @pytest.mark.asyncio
    async def test_get_next_trading_day_skips_weekend(self, market_calendar):
        """Test get_next_trading_day skips weekend."""
        friday = date(2024, 1, 5)

        # Mock is_trading_day to properly detect weekends
        async def mock_is_trading_day(check_date):
            return check_date.weekday() < 5  # Monday-Friday only

        with patch.object(market_calendar, 'is_trading_day', side_effect=mock_is_trading_day):
            next_day = await market_calendar.get_next_trading_day(friday)
            assert next_day == date(2024, 1, 8)  # Monday


# ─────────────────────────────────────────────────────────────────────────────
# TEST MARKET SESSION SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

class TestMarketSessionScheduler:
    """Tests for MarketSessionScheduler class."""

    @pytest.fixture
    def scheduler(self, market_calendar):
        """Create a MarketSessionScheduler instance."""
        return MarketSessionScheduler(market_calendar)

    def test_on_event_registers_callback(self, scheduler):
        """Test callback registration for events."""
        callback = MagicMock()
        scheduler.on_event("market_open", callback)

        assert callback in scheduler._callbacks["market_open"]

    def test_on_event_invalid_event(self, scheduler):
        """Test invalid event doesn't register."""
        callback = MagicMock()
        scheduler.on_event("invalid_event", callback)

        # Should not raise, just ignore
        assert "invalid_event" not in scheduler._callbacks

    @pytest.mark.asyncio
    async def test_start_sets_running(self, scheduler):
        """Test start sets running flag."""
        await scheduler.start()
        assert scheduler._running is True
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_running(self, scheduler):
        """Test stop clears running flag."""
        await scheduler.start()
        await scheduler.stop()
        assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_trigger_event_calls_callbacks(self, scheduler):
        """Test _trigger_event calls registered callbacks."""
        callback = AsyncMock()
        scheduler.on_event("market_open", callback)

        await scheduler._trigger_event("market_open")

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_event_handles_sync_callbacks(self, scheduler):
        """Test _trigger_event handles sync callbacks."""
        callback = MagicMock()
        scheduler.on_event("market_close", callback)

        await scheduler._trigger_event("market_close")

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_event_handles_errors(self, scheduler):
        """Test _trigger_event handles callback errors."""
        callback = AsyncMock(side_effect=Exception("Test error"))
        scheduler.on_event("trading_start", callback)

        # Should not raise
        await scheduler._trigger_event("trading_start")


# ─────────────────────────────────────────────────────────────────────────────
# TEST TIME ZONE HANDLING
# ─────────────────────────────────────────────────────────────────────────────

class TestTimezoneHandling:
    """Tests for timezone handling."""

    def test_calendar_uses_configured_timezone(self, market_calendar):
        """Test calendar uses configured timezone."""
        assert str(market_calendar.timezone) == "Asia/Kolkata"

    def test_now_returns_datetime(self, market_calendar):
        """Test now() returns datetime object."""
        now = market_calendar.now()
        assert isinstance(now, datetime)

    def test_today_returns_date(self, market_calendar):
        """Test today() returns date object."""
        today = market_calendar.today()
        assert isinstance(today, date)


# ─────────────────────────────────────────────────────────────────────────────
# RUN TESTS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
