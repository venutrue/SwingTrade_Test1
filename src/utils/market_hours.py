"""
Market Hours and Trading Calendar
==================================
Production-grade market hours management with holiday calendar.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, date, time, timedelta
from typing import Optional, List, Tuple
from zoneinfo import ZoneInfo
import logging

from src.core.database import db, MarketHolidayRepository
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# NSE 2025-2026 HOLIDAYS (Update annually)
# ─────────────────────────────────────────────────────────────────────────────

NSE_HOLIDAYS_2025 = [
    ("2025-01-26", "Republic Day"),
    ("2025-02-26", "Mahashivratri"),
    ("2025-03-14", "Holi"),
    ("2025-03-31", "Id-Ul-Fitr"),
    ("2025-04-10", "Shri Mahavir Jayanti"),
    ("2025-04-14", "Dr. Ambedkar Jayanti"),
    ("2025-04-18", "Good Friday"),
    ("2025-05-01", "Maharashtra Day"),
    ("2025-06-07", "Bakri Id"),
    ("2025-08-15", "Independence Day"),
    ("2025-08-16", "Parsi New Year"),
    ("2025-08-27", "Ganesh Chaturthi"),
    ("2025-10-02", "Gandhi Jayanti"),
    ("2025-10-21", "Dussehra"),
    ("2025-10-22", "Diwali Laxmi Pujan (Trading Holiday)"),
    ("2025-11-05", "Diwali-Balipratipada"),
    ("2025-11-05", "Prakash Gurpurb Sri Guru Nanak Dev"),
    ("2025-12-25", "Christmas"),
]

NSE_HOLIDAYS_2026 = [
    ("2026-01-26", "Republic Day"),
    ("2026-02-17", "Mahashivratri"),
    ("2026-03-03", "Holi"),
    ("2026-03-20", "Id-Ul-Fitr"),
    ("2026-03-30", "Shri Mahavir Jayanti"),
    ("2026-04-03", "Good Friday"),
    ("2026-04-14", "Dr. Ambedkar Jayanti"),
    ("2026-05-01", "Maharashtra Day"),
    ("2026-05-28", "Bakri Id"),
    ("2026-08-15", "Independence Day"),
    ("2026-08-27", "Janmashtami"),
    ("2026-10-02", "Gandhi Jayanti"),
    ("2026-10-09", "Dussehra"),
    ("2026-10-26", "Diwali Laxmi Pujan"),
    ("2026-10-27", "Diwali-Balipratipada"),
    ("2026-11-25", "Prakash Gurpurb Sri Guru Nanak Dev"),
    ("2026-12-25", "Christmas"),
]


# ─────────────────────────────────────────────────────────────────────────────
# MARKET CALENDAR
# ─────────────────────────────────────────────────────────────────────────────

class MarketCalendar:
    """
    Market calendar with holiday management and trading session checks.
    """

    def __init__(self):
        self.settings = get_settings()
        self.timezone = ZoneInfo(self.settings.market.timezone)
        self._holidays_loaded = False
        self._holidays_cache: set[date] = set()

    async def initialize(self) -> None:
        """Initialize calendar and load holidays into database."""
        if self._holidays_loaded:
            return

        async with db.session() as session:
            repo = MarketHolidayRepository(session)

            # Load 2025 holidays
            for date_str, name in NSE_HOLIDAYS_2025:
                holiday_date = datetime.strptime(date_str, "%Y-%m-%d")
                try:
                    await repo.add_holiday(
                        date=holiday_date,
                        exchange="NSE",
                        name=name,
                    )
                except Exception:
                    pass  # Already exists

            # Load 2026 holidays
            for date_str, name in NSE_HOLIDAYS_2026:
                holiday_date = datetime.strptime(date_str, "%Y-%m-%d")
                try:
                    await repo.add_holiday(
                        date=holiday_date,
                        exchange="NSE",
                        name=name,
                    )
                except Exception:
                    pass

            # Cache holidays
            for year in [2025, 2026]:
                holidays = await repo.get_holidays(year, "NSE")
                for h in holidays:
                    self._holidays_cache.add(h.date.date())

        self._holidays_loaded = True
        logger.info(f"Market calendar initialized with {len(self._holidays_cache)} holidays")

    def now(self) -> datetime:
        """Get current time in market timezone."""
        return datetime.now(self.timezone)

    def today(self) -> date:
        """Get today's date in market timezone."""
        return self.now().date()

    def is_weekend(self, check_date: Optional[date] = None) -> bool:
        """Check if date is a weekend."""
        if check_date is None:
            check_date = self.today()
        return check_date.weekday() >= 5  # Saturday = 5, Sunday = 6

    async def is_holiday(self, check_date: Optional[date] = None) -> bool:
        """Check if date is a market holiday."""
        if check_date is None:
            check_date = self.today()

        # Check cache first
        if check_date in self._holidays_cache:
            return True

        # Query database
        async with db.session() as session:
            repo = MarketHolidayRepository(session)
            return await repo.is_holiday(check_date, self.settings.market.exchange.value)

    async def is_trading_day(self, check_date: Optional[date] = None) -> bool:
        """Check if date is a valid trading day."""
        if check_date is None:
            check_date = self.today()

        if self.is_weekend(check_date):
            return False

        if await self.is_holiday(check_date):
            return False

        return True

    def get_market_open_time(self) -> time:
        """Get market open time."""
        h, m = map(int, self.settings.market.market_open_time.split(":"))
        return time(h, m, 0)

    def get_market_close_time(self) -> time:
        """Get market close time."""
        h, m = map(int, self.settings.market.market_close_time.split(":"))
        return time(h, m, 0)

    def get_trading_start_time(self) -> time:
        """Get trading start time (market open + buffer)."""
        market_open = self.get_market_open_time()
        buffer = self.settings.market.trading_start_buffer_minutes

        dt = datetime.combine(date.today(), market_open)
        dt += timedelta(minutes=buffer)
        return dt.time()

    def get_trading_end_time(self) -> time:
        """Get trading end time (market close - buffer)."""
        market_close = self.get_market_close_time()
        buffer = self.settings.market.trading_end_buffer_minutes

        dt = datetime.combine(date.today(), market_close)
        dt -= timedelta(minutes=buffer)
        return dt.time()

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = self.now()
        current_time = now.time()

        market_open = self.get_market_open_time()
        market_close = self.get_market_close_time()

        return market_open <= current_time <= market_close

    def is_trading_hours(self) -> bool:
        """Check if currently within trading hours (with buffers)."""
        now = self.now()
        current_time = now.time()

        trading_start = self.get_trading_start_time()
        trading_end = self.get_trading_end_time()

        return trading_start <= current_time <= trading_end

    async def can_trade_now(self) -> Tuple[bool, str]:
        """
        Comprehensive check if trading is allowed right now.

        Returns: (can_trade, reason)
        """
        # Check if trading day
        if not await self.is_trading_day():
            if self.is_weekend():
                return False, "Weekend - markets closed"
            return False, "Market holiday"

        # Check if market open
        if not self.is_market_open():
            now = self.now().time()
            market_open = self.get_market_open_time()
            market_close = self.get_market_close_time()

            if now < market_open:
                return False, f"Pre-market: Market opens at {market_open}"
            else:
                return False, f"After-hours: Market closed at {market_close}"

        # Check if within trading window
        if not self.is_trading_hours():
            trading_start = self.get_trading_start_time()
            trading_end = self.get_trading_end_time()
            now = self.now().time()

            if now < trading_start:
                return False, f"Waiting for trading window: Starts at {trading_start}"
            else:
                return False, f"Trading window closed at {trading_end}"

        return True, "Trading allowed"

    def time_until_market_open(self) -> Optional[timedelta]:
        """Get time until market opens (None if already open)."""
        now = self.now()
        current_time = now.time()
        market_open = self.get_market_open_time()

        if current_time >= market_open:
            return None

        market_open_dt = datetime.combine(now.date(), market_open, tzinfo=self.timezone)
        return market_open_dt - now

    def time_until_market_close(self) -> Optional[timedelta]:
        """Get time until market closes (None if already closed)."""
        now = self.now()
        current_time = now.time()
        market_close = self.get_market_close_time()

        if current_time >= market_close:
            return None

        market_close_dt = datetime.combine(now.date(), market_close, tzinfo=self.timezone)
        return market_close_dt - now

    async def get_next_trading_day(self, from_date: Optional[date] = None) -> date:
        """Get the next valid trading day."""
        if from_date is None:
            from_date = self.today()

        check_date = from_date + timedelta(days=1)

        # Check up to 10 days ahead (handles long holiday periods)
        for _ in range(10):
            if await self.is_trading_day(check_date):
                return check_date
            check_date += timedelta(days=1)

        # Fallback - should rarely happen
        return check_date

    def get_session_info(self) -> dict:
        """Get current session information."""
        now = self.now()

        return {
            "current_time": now.isoformat(),
            "timezone": str(self.timezone),
            "market_open": self.get_market_open_time().isoformat(),
            "market_close": self.get_market_close_time().isoformat(),
            "trading_start": self.get_trading_start_time().isoformat(),
            "trading_end": self.get_trading_end_time().isoformat(),
            "is_market_open": self.is_market_open(),
            "is_trading_hours": self.is_trading_hours(),
            "time_until_close": str(self.time_until_market_close()) if self.time_until_market_close() else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# MARKET SESSION SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

class MarketSessionScheduler:
    """
    Scheduler for market session events.
    """

    def __init__(self, calendar: MarketCalendar):
        self.calendar = calendar
        self._callbacks: dict = {
            "pre_market": [],
            "market_open": [],
            "trading_start": [],
            "trading_end": [],
            "market_close": [],
        }
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def on_event(self, event: str, callback) -> None:
        """Register callback for a market event."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info("Market session scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Market session scheduler stopped")

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        last_triggered = {}

        while self._running:
            try:
                now = self.calendar.now()
                current_time = now.time()
                today = now.date()

                # Define events and their times
                events = [
                    ("pre_market", time(9, 0)),  # 9:00 AM
                    ("market_open", self.calendar.get_market_open_time()),
                    ("trading_start", self.calendar.get_trading_start_time()),
                    ("trading_end", self.calendar.get_trading_end_time()),
                    ("market_close", self.calendar.get_market_close_time()),
                ]

                for event_name, event_time in events:
                    # Check if we should trigger this event
                    event_key = f"{today}_{event_name}"
                    if event_key not in last_triggered:
                        # Within 1 minute window of event time
                        event_dt = datetime.combine(today, event_time)
                        now_dt = datetime.combine(today, current_time)
                        diff = (now_dt - event_dt).total_seconds()

                        if 0 <= diff < 60:  # Within first minute after event
                            if await self.calendar.is_trading_day():
                                await self._trigger_event(event_name)
                                last_triggered[event_key] = now

                # Clean old entries
                yesterday = today - timedelta(days=1)
                last_triggered = {
                    k: v for k, v in last_triggered.items()
                    if not k.startswith(str(yesterday))
                }

                # Sleep for 30 seconds
                await asyncio.sleep(30)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)

    async def _trigger_event(self, event_name: str) -> None:
        """Trigger callbacks for an event."""
        logger.info(f"Market event: {event_name}")

        for callback in self._callbacks[event_name]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Error in {event_name} callback: {e}")


# Global instance
market_calendar = MarketCalendar()
