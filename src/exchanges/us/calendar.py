"""
US Market Calendar
==================
NYSE/NASDAQ market hours and holidays.
"""

from __future__ import annotations

from datetime import date, time, datetime
from typing import Dict, List, Optional
import logging

from src.exchanges.base import BaseMarketCalendar, ExchangeCode

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# US MARKET HOLIDAYS
# ─────────────────────────────────────────────────────────────────────────────

# NYSE/NASDAQ Holidays 2024-2026
# Source: https://www.nyse.com/markets/hours-calendars

US_HOLIDAYS = {
    2024: [
        (date(2024, 1, 1), "New Year's Day"),
        (date(2024, 1, 15), "Martin Luther King Jr. Day"),
        (date(2024, 2, 19), "Presidents Day"),
        (date(2024, 3, 29), "Good Friday"),
        (date(2024, 5, 27), "Memorial Day"),
        (date(2024, 6, 19), "Juneteenth"),
        (date(2024, 7, 4), "Independence Day"),
        (date(2024, 9, 2), "Labor Day"),
        (date(2024, 11, 28), "Thanksgiving Day"),
        (date(2024, 12, 25), "Christmas Day"),
    ],
    2025: [
        (date(2025, 1, 1), "New Year's Day"),
        (date(2025, 1, 20), "Martin Luther King Jr. Day"),
        (date(2025, 2, 17), "Presidents Day"),
        (date(2025, 4, 18), "Good Friday"),
        (date(2025, 5, 26), "Memorial Day"),
        (date(2025, 6, 19), "Juneteenth"),
        (date(2025, 7, 4), "Independence Day"),
        (date(2025, 9, 1), "Labor Day"),
        (date(2025, 11, 27), "Thanksgiving Day"),
        (date(2025, 12, 25), "Christmas Day"),
    ],
    2026: [
        (date(2026, 1, 1), "New Year's Day"),
        (date(2026, 1, 19), "Martin Luther King Jr. Day"),
        (date(2026, 2, 16), "Presidents Day"),
        (date(2026, 4, 3), "Good Friday"),
        (date(2026, 5, 25), "Memorial Day"),
        (date(2026, 6, 19), "Juneteenth"),
        (date(2026, 7, 3), "Independence Day (observed)"),
        (date(2026, 9, 7), "Labor Day"),
        (date(2026, 11, 26), "Thanksgiving Day"),
        (date(2026, 12, 25), "Christmas Day"),
    ],
}

# Early close days (1:00 PM ET)
US_EARLY_CLOSES = {
    2024: [
        date(2024, 7, 3),    # Day before Independence Day
        date(2024, 11, 29),  # Day after Thanksgiving
        date(2024, 12, 24),  # Christmas Eve
    ],
    2025: [
        date(2025, 7, 3),    # Day before Independence Day
        date(2025, 11, 28),  # Day after Thanksgiving
        date(2025, 12, 24),  # Christmas Eve
    ],
    2026: [
        date(2026, 11, 27),  # Day after Thanksgiving
        date(2026, 12, 24),  # Christmas Eve
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# US MARKET CALENDAR
# ─────────────────────────────────────────────────────────────────────────────

class USMarketCalendar(BaseMarketCalendar):
    """
    Market calendar for NYSE/NASDAQ.

    Regular hours: 9:30 AM - 4:00 PM Eastern Time
    Pre-market: 4:00 AM - 9:30 AM ET
    After-hours: 4:00 PM - 8:00 PM ET
    """

    def __init__(self, exchange: ExchangeCode = ExchangeCode.NYSE):
        if exchange not in (ExchangeCode.NYSE, ExchangeCode.NASDAQ):
            raise ValueError(f"USMarketCalendar only supports NYSE/NASDAQ, got {exchange}")
        super().__init__(exchange)
        self._holiday_cache: Dict[int, set] = {}

    async def get_holidays(self, year: int) -> List[date]:
        """Get list of market holidays for a year."""
        if year in US_HOLIDAYS:
            return [h[0] for h in US_HOLIDAYS[year]]

        # For years not explicitly listed, use typical US holiday schedule
        # This is a fallback - should update the calendar yearly
        logger.warning(f"Holiday calendar for {year} not available, using estimate")
        return self._estimate_holidays(year)

    async def is_holiday(self, check_date: date) -> bool:
        """Check if a date is a market holiday."""
        year = check_date.year

        # Build cache if not exists
        if year not in self._holiday_cache:
            holidays = await self.get_holidays(year)
            self._holiday_cache[year] = set(holidays)

        return check_date in self._holiday_cache[year]

    async def get_early_closes(self, year: int) -> Dict[date, time]:
        """Get early close dates and times."""
        early_close_time = time(13, 0)  # 1:00 PM ET

        if year in US_EARLY_CLOSES:
            return {d: early_close_time for d in US_EARLY_CLOSES[year]}

        return {}

    def _estimate_holidays(self, year: int) -> List[date]:
        """Estimate holidays for years without explicit calendar."""
        from calendar import Calendar

        holidays = []

        # New Year's Day (Jan 1, or nearest weekday)
        holidays.append(self._adjust_for_weekend(date(year, 1, 1)))

        # MLK Day (3rd Monday of January)
        holidays.append(self._nth_weekday(year, 1, 0, 3))

        # Presidents Day (3rd Monday of February)
        holidays.append(self._nth_weekday(year, 2, 0, 3))

        # Good Friday (variable - roughly late March/April)
        # This is complex to calculate, skip for estimate

        # Memorial Day (last Monday of May)
        holidays.append(self._last_weekday(year, 5, 0))

        # Juneteenth (June 19, or nearest weekday)
        holidays.append(self._adjust_for_weekend(date(year, 6, 19)))

        # Independence Day (July 4, or nearest weekday)
        holidays.append(self._adjust_for_weekend(date(year, 7, 4)))

        # Labor Day (1st Monday of September)
        holidays.append(self._nth_weekday(year, 9, 0, 1))

        # Thanksgiving (4th Thursday of November)
        holidays.append(self._nth_weekday(year, 11, 3, 4))

        # Christmas (Dec 25, or nearest weekday)
        holidays.append(self._adjust_for_weekend(date(year, 12, 25)))

        return holidays

    def _adjust_for_weekend(self, d: date) -> date:
        """Adjust holiday to nearest weekday if on weekend."""
        if d.weekday() == 5:  # Saturday
            return d - timedelta(days=1)
        elif d.weekday() == 6:  # Sunday
            return d + timedelta(days=1)
        return d

    def _nth_weekday(self, year: int, month: int, weekday: int, n: int) -> date:
        """Get nth occurrence of a weekday in a month."""
        from calendar import monthcalendar

        cal = monthcalendar(year, month)
        count = 0

        for week in cal:
            if week[weekday] != 0:
                count += 1
                if count == n:
                    return date(year, month, week[weekday])

        raise ValueError(f"Could not find {n}th weekday {weekday} in {year}-{month}")

    def _last_weekday(self, year: int, month: int, weekday: int) -> date:
        """Get last occurrence of a weekday in a month."""
        from calendar import monthcalendar

        cal = monthcalendar(year, month)

        for week in reversed(cal):
            if week[weekday] != 0:
                return date(year, month, week[weekday])

        raise ValueError(f"Could not find last weekday {weekday} in {year}-{month}")

    def is_extended_hours(self) -> bool:
        """Check if currently in extended hours (pre/post market)."""
        now = self.now()
        current_time = now.time()

        # Pre-market
        if self.config.pre_market_start <= current_time < self.config.market_open:
            return True

        # Post-market
        if self.config.market_close < current_time <= self.config.post_market_end:
            return True

        return False

    def get_session_type(self) -> str:
        """Get current session type."""
        now = self.now()
        current_time = now.time()

        if current_time < self.config.pre_market_start:
            return "CLOSED"
        elif current_time < self.config.market_open:
            return "PRE_MARKET"
        elif current_time <= self.config.market_close:
            return "REGULAR"
        elif current_time <= self.config.post_market_end:
            return "POST_MARKET"
        else:
            return "CLOSED"


# Convenience function
def get_us_calendar(exchange: ExchangeCode = ExchangeCode.NYSE) -> USMarketCalendar:
    return USMarketCalendar(exchange)
