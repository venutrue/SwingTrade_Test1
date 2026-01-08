"""
Yahoo Finance Data Provider
============================
Free historical market data from Yahoo Finance.

Supports both US and Indian markets.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List
import logging
import aiohttp

import pandas as pd
import numpy as np

from src.exchanges.base import BaseDataProvider, ExchangeCode

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# YAHOO FINANCE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

YAHOO_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
YAHOO_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"

# Symbol suffix mapping for different exchanges
EXCHANGE_SUFFIX = {
    ExchangeCode.NSE: ".NS",
    ExchangeCode.BSE: ".BO",
    ExchangeCode.NYSE: "",
    ExchangeCode.NASDAQ: "",
}

# Interval mapping
INTERVAL_MAP = {
    "1min": "1m",
    "2min": "2m",
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
    "60min": "60m",
    "1hour": "1h",
    "1day": "1d",
    "day": "1d",
    "1week": "1wk",
    "week": "1wk",
    "1month": "1mo",
    "month": "1mo",
}


# ─────────────────────────────────────────────────────────────────────────────
# YAHOO FINANCE DATA PROVIDER
# ─────────────────────────────────────────────────────────────────────────────

class YahooFinanceProvider(BaseDataProvider):
    """
    Yahoo Finance data provider.

    Features:
    - Free historical data (daily, weekly, monthly)
    - Intraday data (limited to recent 7 days for free)
    - Supports US and Indian markets
    - Adjusted prices for splits/dividends
    """

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session exists."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def close(self) -> None:
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _format_symbol(
        self,
        symbol: str,
        exchange: Optional[ExchangeCode] = None,
    ) -> str:
        """Format symbol for Yahoo Finance API."""
        # If already has suffix, return as-is
        if "." in symbol:
            return symbol

        # Add exchange suffix
        if exchange:
            suffix = EXCHANGE_SUFFIX.get(exchange, "")
            return f"{symbol}{suffix}"

        # Try to detect from symbol
        return symbol

    def supports_exchange(self, exchange: ExchangeCode) -> bool:
        """Check if provider supports an exchange."""
        return exchange in EXCHANGE_SUFFIX

    async def get_historical_data(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        exchange: Optional[ExchangeCode] = None,
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data.

        Args:
            symbol: Stock symbol (e.g., "AAPL", "RELIANCE")
            interval: Data interval ("1day", "1hour", "5min", etc.)
            start: Start datetime
            end: End datetime
            exchange: Exchange code (helps format symbol correctly)

        Returns:
            DataFrame with columns: [open, high, low, close, volume]
        """
        session = await self._ensure_session()

        yahoo_symbol = self._format_symbol(symbol, exchange)
        yahoo_interval = INTERVAL_MAP.get(interval.lower(), "1d")

        # Calculate period
        period1 = int(start.timestamp())
        period2 = int(end.timestamp())

        params = {
            "period1": period1,
            "period2": period2,
            "interval": yahoo_interval,
            "includePrePost": "false",
            "events": "div,splits",
        }

        url = f"{YAHOO_BASE_URL}/{yahoo_symbol}"

        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Yahoo Finance error {response.status} for {yahoo_symbol}")
                    return pd.DataFrame()

                data = await response.json()

            # Parse response
            result = data.get("chart", {}).get("result", [])
            if not result:
                logger.warning(f"No data returned for {yahoo_symbol}")
                return pd.DataFrame()

            result = result[0]
            timestamps = result.get("timestamp", [])
            quote = result.get("indicators", {}).get("quote", [{}])[0]
            adjclose = result.get("indicators", {}).get("adjclose", [{}])

            if not timestamps:
                return pd.DataFrame()

            # Build DataFrame
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(timestamps, unit="s"),
                "open": quote.get("open", []),
                "high": quote.get("high", []),
                "low": quote.get("low", []),
                "close": quote.get("close", []),
                "volume": quote.get("volume", []),
            })

            # Use adjusted close if available
            if adjclose and adjclose[0].get("adjclose"):
                df["adj_close"] = adjclose[0]["adjclose"]

            df.set_index("timestamp", inplace=True)
            df = df.dropna()

            # Convert to proper types
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)

            logger.info(f"Fetched {len(df)} bars for {yahoo_symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch data for {yahoo_symbol}: {e}")
            return pd.DataFrame()

    async def get_quote(
        self,
        symbol: str,
        exchange: Optional[ExchangeCode] = None,
    ) -> Dict[str, Any]:
        """Get real-time quote (delayed by ~15 minutes)."""
        session = await self._ensure_session()

        yahoo_symbol = self._format_symbol(symbol, exchange)
        url = f"{YAHOO_BASE_URL}/{yahoo_symbol}"

        params = {
            "interval": "1d",
            "range": "1d",
        }

        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return {}

                data = await response.json()

            result = data.get("chart", {}).get("result", [])
            if not result:
                return {}

            meta = result[0].get("meta", {})

            return {
                "symbol": symbol,
                "price": Decimal(str(meta.get("regularMarketPrice", 0))),
                "previous_close": Decimal(str(meta.get("previousClose", 0))),
                "open": Decimal(str(meta.get("regularMarketOpen", 0))),
                "day_high": Decimal(str(meta.get("regularMarketDayHigh", 0))),
                "day_low": Decimal(str(meta.get("regularMarketDayLow", 0))),
                "volume": meta.get("regularMarketVolume", 0),
                "market_cap": meta.get("marketCap"),
                "currency": meta.get("currency"),
                "exchange": meta.get("exchangeName"),
            }

        except Exception as e:
            logger.error(f"Failed to get quote for {yahoo_symbol}: {e}")
            return {}

    async def search_symbols(
        self,
        query: str,
        exchange: Optional[ExchangeCode] = None,
    ) -> List[Dict[str, Any]]:
        """Search for symbols."""
        session = await self._ensure_session()

        params = {
            "q": query,
            "quotesCount": 10,
            "newsCount": 0,
        }

        try:
            async with session.get(YAHOO_SEARCH_URL, params=params) as response:
                if response.status != 200:
                    return []

                data = await response.json()

            quotes = data.get("quotes", [])
            results = []

            for q in quotes:
                # Filter by exchange if specified
                if exchange:
                    exch = q.get("exchange", "")
                    if exchange == ExchangeCode.NSE and "NSE" not in exch:
                        continue
                    if exchange == ExchangeCode.NYSE and exch not in ("NYQ", "NYSE"):
                        continue
                    if exchange == ExchangeCode.NASDAQ and exch not in ("NMS", "NASDAQ"):
                        continue

                results.append({
                    "symbol": q.get("symbol", "").replace(".NS", "").replace(".BO", ""),
                    "name": q.get("longname") or q.get("shortname", ""),
                    "exchange": q.get("exchange"),
                    "type": q.get("typeDisp"),
                    "yahoo_symbol": q.get("symbol"),
                })

            return results

        except Exception as e:
            logger.error(f"Symbol search failed: {e}")
            return []

    async def get_multiple_symbols(
        self,
        symbols: List[str],
        interval: str,
        start: datetime,
        end: datetime,
        exchange: Optional[ExchangeCode] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols.

        Returns dict of symbol -> DataFrame.
        """
        tasks = [
            self.get_historical_data(symbol, interval, start, end, exchange)
            for symbol in symbols
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, pd.DataFrame) and not result.empty:
                data[symbol] = result
            elif isinstance(result, Exception):
                logger.error(f"Error fetching {symbol}: {result}")

        return data


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

async def download_historical_data(
    symbols: List[str],
    start: datetime,
    end: datetime,
    interval: str = "1day",
    exchange: Optional[ExchangeCode] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Download historical data for multiple symbols.

    Example:
        data = await download_historical_data(
            symbols=["AAPL", "MSFT", "GOOGL"],
            start=datetime(2020, 1, 1),
            end=datetime(2024, 1, 1),
            exchange=ExchangeCode.NYSE,
        )
    """
    provider = YahooFinanceProvider()
    try:
        return await provider.get_multiple_symbols(
            symbols, interval, start, end, exchange
        )
    finally:
        await provider.close()


async def get_stock_data(
    symbol: str,
    years: int = 5,
    exchange: Optional[ExchangeCode] = None,
) -> pd.DataFrame:
    """
    Get historical data for a single stock.

    Example:
        df = await get_stock_data("RELIANCE", years=5, exchange=ExchangeCode.NSE)
    """
    provider = YahooFinanceProvider()
    try:
        end = datetime.now()
        start = end - timedelta(days=years * 365)
        return await provider.get_historical_data(
            symbol, "1day", start, end, exchange
        )
    finally:
        await provider.close()


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE DATA FOR TESTING
# ─────────────────────────────────────────────────────────────────────────────

# Blue chip symbols for each market
US_BLUE_CHIPS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet
    "AMZN",   # Amazon
    "JPM",    # JPMorgan Chase
    "V",      # Visa
    "JNJ",    # Johnson & Johnson
    "WMT",    # Walmart
    "PG",     # Procter & Gamble
    "MA",     # Mastercard
    "HD",     # Home Depot
    "DIS",    # Disney
    "NVDA",   # NVIDIA
    "BAC",    # Bank of America
    "KO",     # Coca-Cola
]

INDIA_BLUE_CHIPS = [
    "RELIANCE",   # Reliance Industries
    "TCS",        # Tata Consultancy Services
    "HDFCBANK",   # HDFC Bank
    "INFY",       # Infosys
    "ICICIBANK",  # ICICI Bank
    "HINDUNILVR", # Hindustan Unilever
    "ITC",        # ITC Limited
    "SBIN",       # State Bank of India
    "BHARTIARTL", # Bharti Airtel
    "KOTAKBANK",  # Kotak Mahindra Bank
    "LT",         # Larsen & Toubro
    "AXISBANK",   # Axis Bank
    "BAJFINANCE", # Bajaj Finance
    "ASIANPAINT", # Asian Paints
    "MARUTI",     # Maruti Suzuki
]
