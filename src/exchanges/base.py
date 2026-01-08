"""
Abstract Exchange Interface
============================
Base classes for exchange-agnostic trading system.
All exchange implementations must conform to these interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
from zoneinfo import ZoneInfo

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# EXCHANGE IDENTIFICATION
# ─────────────────────────────────────────────────────────────────────────────

class ExchangeCode(str, Enum):
    """Supported exchanges."""
    # India
    NSE = "NSE"
    BSE = "BSE"

    # United States
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    AMEX = "AMEX"

    # Futures & Options
    NFO = "NFO"      # NSE F&O
    CME = "CME"      # Chicago Mercantile


class Currency(str, Enum):
    """Supported currencies."""
    INR = "INR"
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"


class Market(str, Enum):
    """Market regions."""
    INDIA = "INDIA"
    US = "US"
    EU = "EU"
    UK = "UK"


# ─────────────────────────────────────────────────────────────────────────────
# EXCHANGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExchangeConfig:
    """Configuration for an exchange."""
    code: ExchangeCode
    name: str
    market: Market
    currency: Currency
    timezone: str

    # Market hours (local time)
    market_open: time
    market_close: time

    # Trading buffers
    open_buffer_minutes: int = 15
    close_buffer_minutes: int = 15

    # Pre/post market
    pre_market_start: Optional[time] = None
    pre_market_end: Optional[time] = None
    post_market_start: Optional[time] = None
    post_market_end: Optional[time] = None

    # Lot sizes
    default_lot_size: int = 1

    # Symbol format
    symbol_prefix: str = ""
    symbol_suffix: str = ""

    def get_timezone(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


# Pre-configured exchanges
EXCHANGE_CONFIGS: Dict[ExchangeCode, ExchangeConfig] = {
    ExchangeCode.NSE: ExchangeConfig(
        code=ExchangeCode.NSE,
        name="National Stock Exchange of India",
        market=Market.INDIA,
        currency=Currency.INR,
        timezone="Asia/Kolkata",
        market_open=time(9, 15),
        market_close=time(15, 30),
        pre_market_start=time(9, 0),
        pre_market_end=time(9, 8),
    ),
    ExchangeCode.NYSE: ExchangeConfig(
        code=ExchangeCode.NYSE,
        name="New York Stock Exchange",
        market=Market.US,
        currency=Currency.USD,
        timezone="America/New_York",
        market_open=time(9, 30),
        market_close=time(16, 0),
        pre_market_start=time(4, 0),
        pre_market_end=time(9, 30),
        post_market_start=time(16, 0),
        post_market_end=time(20, 0),
    ),
    ExchangeCode.NASDAQ: ExchangeConfig(
        code=ExchangeCode.NASDAQ,
        name="NASDAQ Stock Exchange",
        market=Market.US,
        currency=Currency.USD,
        timezone="America/New_York",
        market_open=time(9, 30),
        market_close=time(16, 0),
        pre_market_start=time(4, 0),
        pre_market_end=time(9, 30),
        post_market_start=time(16, 0),
        post_market_end=time(20, 0),
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# FEE STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeeStructure:
    """Trading fee structure for an exchange/broker combination."""
    # Per-order fees
    commission_per_order: Decimal = Decimal("0")
    commission_per_share: Decimal = Decimal("0")
    commission_percent: Decimal = Decimal("0")
    min_commission: Decimal = Decimal("0")
    max_commission: Optional[Decimal] = None

    # Regulatory fees
    exchange_fee_percent: Decimal = Decimal("0")
    regulatory_fee_percent: Decimal = Decimal("0")
    transaction_tax_percent: Decimal = Decimal("0")  # STT in India, SEC fee in US

    # Other
    clearing_fee_percent: Decimal = Decimal("0")

    def calculate_total_fee(
        self,
        trade_value: Decimal,
        quantity: int,
        side: str,
    ) -> Decimal:
        """Calculate total fees for a trade."""
        # Commission
        commission = max(
            self.min_commission,
            self.commission_per_order +
            (self.commission_per_share * quantity) +
            (trade_value * self.commission_percent / 100)
        )
        if self.max_commission:
            commission = min(commission, self.max_commission)

        # Regulatory fees
        regulatory = trade_value * (
            self.exchange_fee_percent +
            self.regulatory_fee_percent +
            self.clearing_fee_percent
        ) / 100

        # Transaction tax (often only on sell side)
        tax = Decimal("0")
        if side == "SELL":
            tax = trade_value * self.transaction_tax_percent / 100

        return commission + regulatory + tax


# Pre-configured fee structures
FEE_STRUCTURES: Dict[str, FeeStructure] = {
    "NSE_ZERODHA": FeeStructure(
        commission_per_order=Decimal("20"),
        transaction_tax_percent=Decimal("0.1"),  # STT
        exchange_fee_percent=Decimal("0.00345"),
        regulatory_fee_percent=Decimal("0.0002"),  # SEBI
        clearing_fee_percent=Decimal("0.00025"),
    ),
    "US_ALPACA": FeeStructure(
        # Alpaca is commission-free
        commission_per_order=Decimal("0"),
        regulatory_fee_percent=Decimal("0.000008"),  # SEC fee
        exchange_fee_percent=Decimal("0.000119"),    # FINRA TAF
    ),
    "US_IBKR_PRO": FeeStructure(
        commission_per_share=Decimal("0.005"),
        min_commission=Decimal("1.0"),
        max_commission=Decimal("0.5"),  # 0.5% of trade value
        regulatory_fee_percent=Decimal("0.000008"),
        exchange_fee_percent=Decimal("0.0003"),
    ),
    "US_IBKR_LITE": FeeStructure(
        # IBKR Lite is commission-free for US stocks
        commission_per_order=Decimal("0"),
        regulatory_fee_percent=Decimal("0.000008"),
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT BROKER INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class BaseBrokerInterface(ABC):
    """
    Abstract interface that all broker implementations must follow.

    This ensures consistent behavior across different brokers/exchanges.
    """

    def __init__(self, exchange: ExchangeCode, fee_structure: FeeStructure):
        self.exchange = exchange
        self.exchange_config = EXCHANGE_CONFIGS[exchange]
        self.fee_structure = fee_structure
        self._connected = False

    # ─────────────────────────────────────────────────────────────────────────
    # CONNECTION MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the broker. Returns True if successful."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the broker."""
        pass

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Check if connection is healthy."""
        pass

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ─────────────────────────────────────────────────────────────────────────
    # ACCOUNT INFORMATION
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    async def get_account(self) -> Dict[str, Any]:
        """Get account information."""
        pass

    @abstractmethod
    async def get_balance(self) -> Decimal:
        """Get available cash balance."""
        pass

    @abstractmethod
    async def get_buying_power(self) -> Decimal:
        """Get buying power (including margin if applicable)."""
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # ORDER MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: str,  # "BUY" or "SELL"
        quantity: int,
        order_type: str = "MARKET",  # "MARKET", "LIMIT", "STOP", "STOP_LIMIT"
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        time_in_force: str = "DAY",  # "DAY", "GTC", "IOC", "FOK"
        extended_hours: bool = False,
    ) -> Dict[str, Any]:
        """
        Place an order.

        Returns dict with at minimum: {"order_id": str, "status": str}
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order."""
        pass

    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order details by ID."""
        pass

    @abstractmethod
    async def get_orders(
        self,
        status: Optional[str] = None,  # "open", "closed", "all"
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get list of orders."""
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # POSITION MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions."""
        pass

    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position for a specific symbol."""
        pass

    @abstractmethod
    async def close_position(
        self,
        symbol: str,
        quantity: Optional[int] = None,  # None = close all
    ) -> Dict[str, Any]:
        """Close a position."""
        pass

    @abstractmethod
    async def close_all_positions(self) -> List[Dict[str, Any]]:
        """Close all positions. Returns list of close orders."""
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # MARKET DATA
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get current quote for a symbol.

        Returns dict with at minimum:
        {"bid": Decimal, "ask": Decimal, "last": Decimal, "volume": int}
        """
        pass

    @abstractmethod
    async def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get quotes for multiple symbols."""
        pass

    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        interval: str,  # "1min", "5min", "15min", "1hour", "1day"
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data.

        Returns DataFrame with columns: [open, high, low, close, volume]
        Index should be datetime.
        """
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # SYMBOL INFORMATION
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    async def get_tradable_symbols(self) -> List[str]:
        """Get list of tradable symbols."""
        pass

    @abstractmethod
    async def is_tradable(self, symbol: str) -> bool:
        """Check if a symbol is tradable."""
        pass

    @abstractmethod
    async def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get symbol information.

        Returns dict with: name, exchange, lot_size, tick_size, etc.
        """
        pass


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT MARKET CALENDAR
# ─────────────────────────────────────────────────────────────────────────────

class BaseMarketCalendar(ABC):
    """Abstract market calendar interface."""

    def __init__(self, exchange: ExchangeCode):
        self.exchange = exchange
        self.config = EXCHANGE_CONFIGS[exchange]
        self.tz = self.config.get_timezone()

    @abstractmethod
    async def get_holidays(self, year: int) -> List[date]:
        """Get list of market holidays for a year."""
        pass

    @abstractmethod
    async def is_holiday(self, check_date: date) -> bool:
        """Check if a date is a market holiday."""
        pass

    @abstractmethod
    async def get_early_closes(self, year: int) -> Dict[date, time]:
        """Get early close dates and times."""
        pass

    def is_weekend(self, check_date: date) -> bool:
        """Check if date is a weekend."""
        return check_date.weekday() >= 5

    async def is_trading_day(self, check_date: date) -> bool:
        """Check if date is a valid trading day."""
        if self.is_weekend(check_date):
            return False
        return not await self.is_holiday(check_date)

    def now(self) -> datetime:
        """Get current time in exchange timezone."""
        return datetime.now(self.tz)

    def today(self) -> date:
        """Get today's date in exchange timezone."""
        return self.now().date()

    async def get_market_close_time(self, check_date: date) -> time:
        """Get market close time, accounting for early closes."""
        early_closes = await self.get_early_closes(check_date.year)
        return early_closes.get(check_date, self.config.market_close)

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = self.now()
        current_time = now.time()
        return self.config.market_open <= current_time <= self.config.market_close

    def is_trading_hours(self) -> bool:
        """Check if within trading hours (with buffers)."""
        now = self.now()
        current_time = now.time()

        # Add buffer to open
        open_dt = datetime.combine(now.date(), self.config.market_open)
        open_dt += timedelta(minutes=self.config.open_buffer_minutes)

        # Subtract buffer from close
        close_dt = datetime.combine(now.date(), self.config.market_close)
        close_dt -= timedelta(minutes=self.config.close_buffer_minutes)

        return open_dt.time() <= current_time <= close_dt.time()

    async def can_trade_now(self) -> Tuple[bool, str]:
        """Comprehensive check if trading is allowed now."""
        if not await self.is_trading_day(self.today()):
            if self.is_weekend(self.today()):
                return False, "Weekend - market closed"
            return False, "Market holiday"

        if not self.is_market_open():
            now = self.now().time()
            if now < self.config.market_open:
                return False, f"Pre-market: Opens at {self.config.market_open}"
            return False, f"After-hours: Closed at {self.config.market_close}"

        if not self.is_trading_hours():
            return False, "Outside trading window (buffer period)"

        return True, "Trading allowed"

    async def get_next_trading_day(self, from_date: Optional[date] = None) -> date:
        """Get the next valid trading day."""
        check_date = (from_date or self.today()) + timedelta(days=1)

        for _ in range(14):  # Check up to 2 weeks
            if await self.is_trading_day(check_date):
                return check_date
            check_date += timedelta(days=1)

        return check_date


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT DATA PROVIDER
# ─────────────────────────────────────────────────────────────────────────────

class BaseDataProvider(ABC):
    """Abstract interface for market data providers."""

    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        exchange: Optional[ExchangeCode] = None,
    ) -> pd.DataFrame:
        """Get historical OHLCV data."""
        pass

    @abstractmethod
    async def get_quote(
        self,
        symbol: str,
        exchange: Optional[ExchangeCode] = None,
    ) -> Dict[str, Any]:
        """Get real-time quote."""
        pass

    @abstractmethod
    async def search_symbols(
        self,
        query: str,
        exchange: Optional[ExchangeCode] = None,
    ) -> List[Dict[str, Any]]:
        """Search for symbols."""
        pass

    @abstractmethod
    def supports_exchange(self, exchange: ExchangeCode) -> bool:
        """Check if provider supports an exchange."""
        pass
