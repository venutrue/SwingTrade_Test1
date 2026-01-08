"""
Alpaca Broker Implementation
============================
Production-grade Alpaca integration for US stock trading.

Alpaca offers:
- Commission-free trading
- Paper trading for testing
- Real-time market data
- RESTful API + WebSocket
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List
import logging
import aiohttp

import pandas as pd

from src.exchanges.base import (
    BaseBrokerInterface, ExchangeCode, FeeStructure, FEE_STRUCTURES
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ALPACA API CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_URL = "https://api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"


# ─────────────────────────────────────────────────────────────────────────────
# ALPACA BROKER
# ─────────────────────────────────────────────────────────────────────────────

class AlpacaBroker(BaseBrokerInterface):
    """
    Alpaca broker implementation for US stock trading.

    Features:
    - Commission-free equity trading
    - Paper trading mode for testing
    - Real-time and historical market data
    - Fractional shares support
    - Extended hours trading
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        paper: bool = True,
        exchange: ExchangeCode = ExchangeCode.NYSE,
    ):
        super().__init__(
            exchange=exchange,
            fee_structure=FEE_STRUCTURES["US_ALPACA"],
        )

        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper

        self.base_url = ALPACA_PAPER_URL if paper else ALPACA_LIVE_URL
        self.data_url = ALPACA_DATA_URL

        self._session: Optional[aiohttp.ClientSession] = None
        self._account_cache: Optional[Dict] = None
        self._account_cache_time: Optional[datetime] = None

    # ─────────────────────────────────────────────────────────────────────────
    # CONNECTION MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
        }

    async def connect(self) -> bool:
        """Connect to Alpaca API."""
        try:
            self._session = aiohttp.ClientSession(
                headers=self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=30),
            )

            # Verify connection by fetching account
            account = await self.get_account()
            if account:
                self._connected = True
                mode = "PAPER" if self.paper else "LIVE"
                logger.info(f"Connected to Alpaca ({mode}): {account.get('id', 'Unknown')}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Alpaca API."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        logger.info("Disconnected from Alpaca")

    async def is_healthy(self) -> bool:
        """Check API health."""
        try:
            account = await self.get_account()
            return account is not None and account.get("status") == "ACTIVE"
        except Exception:
            return False

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        base_url: Optional[str] = None,
    ) -> Optional[Dict]:
        """Make API request."""
        if not self._session:
            raise RuntimeError("Not connected to Alpaca")

        url = f"{base_url or self.base_url}{endpoint}"

        try:
            async with self._session.request(
                method,
                url,
                json=data,
                params=params,
            ) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 204:
                    return {}
                else:
                    error_text = await response.text()
                    logger.error(f"Alpaca API error {response.status}: {error_text}")
                    return None

        except Exception as e:
            logger.error(f"Alpaca request failed: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # ACCOUNT INFORMATION
    # ─────────────────────────────────────────────────────────────────────────

    async def get_account(self) -> Dict[str, Any]:
        """Get account information."""
        # Cache for 5 seconds to reduce API calls
        now = datetime.now()
        if (
            self._account_cache
            and self._account_cache_time
            and (now - self._account_cache_time).seconds < 5
        ):
            return self._account_cache

        result = await self._request("GET", "/v2/account")
        if result:
            self._account_cache = result
            self._account_cache_time = now

        return result or {}

    async def get_balance(self) -> Decimal:
        """Get available cash balance."""
        account = await self.get_account()
        return Decimal(str(account.get("cash", "0")))

    async def get_buying_power(self) -> Decimal:
        """Get buying power."""
        account = await self.get_account()
        return Decimal(str(account.get("buying_power", "0")))

    # ─────────────────────────────────────────────────────────────────────────
    # ORDER MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        time_in_force: str = "DAY",
        extended_hours: bool = False,
    ) -> Dict[str, Any]:
        """Place an order."""
        # Map order type
        alpaca_type = {
            "MARKET": "market",
            "LIMIT": "limit",
            "STOP": "stop",
            "STOP_LIMIT": "stop_limit",
        }.get(order_type.upper(), "market")

        # Map time in force
        alpaca_tif = {
            "DAY": "day",
            "GTC": "gtc",
            "IOC": "ioc",
            "FOK": "fok",
            "OPG": "opg",  # Market on open
            "CLS": "cls",  # Market on close
        }.get(time_in_force.upper(), "day")

        order_data = {
            "symbol": symbol.upper(),
            "qty": str(quantity),
            "side": side.lower(),
            "type": alpaca_type,
            "time_in_force": alpaca_tif,
        }

        if price and alpaca_type in ("limit", "stop_limit"):
            order_data["limit_price"] = str(price)

        if stop_price and alpaca_type in ("stop", "stop_limit"):
            order_data["stop_price"] = str(stop_price)

        if extended_hours:
            order_data["extended_hours"] = True

        result = await self._request("POST", "/v2/orders", data=order_data)

        if result:
            return {
                "status": "success",
                "order_id": result.get("id"),
                "client_order_id": result.get("client_order_id"),
                "symbol": result.get("symbol"),
                "side": result.get("side"),
                "quantity": int(result.get("qty", 0)),
                "order_type": result.get("type"),
                "order_status": result.get("status"),
                "filled_qty": int(float(result.get("filled_qty", 0))),
                "filled_avg_price": result.get("filled_avg_price"),
                "created_at": result.get("created_at"),
            }

        return {"status": "error", "error": "Order placement failed"}

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order."""
        result = await self._request("DELETE", f"/v2/orders/{order_id}")

        if result is not None:  # 204 returns empty dict
            return {"status": "success", "order_id": order_id}

        return {"status": "error", "error": "Cancel failed"}

    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order by ID."""
        return await self._request("GET", f"/v2/orders/{order_id}")

    async def get_orders(
        self,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get list of orders."""
        params = {"limit": limit}

        if status:
            params["status"] = status.lower()

        result = await self._request("GET", "/v2/orders", params=params)
        return result if isinstance(result, list) else []

    # ─────────────────────────────────────────────────────────────────────────
    # POSITION MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions."""
        result = await self._request("GET", "/v2/positions")

        if isinstance(result, list):
            return [
                {
                    "symbol": p.get("symbol"),
                    "quantity": int(p.get("qty", 0)),
                    "side": "long" if int(p.get("qty", 0)) > 0 else "short",
                    "avg_entry_price": Decimal(str(p.get("avg_entry_price", 0))),
                    "market_value": Decimal(str(p.get("market_value", 0))),
                    "cost_basis": Decimal(str(p.get("cost_basis", 0))),
                    "unrealized_pl": Decimal(str(p.get("unrealized_pl", 0))),
                    "unrealized_plpc": Decimal(str(p.get("unrealized_plpc", 0))),
                    "current_price": Decimal(str(p.get("current_price", 0))),
                }
                for p in result
            ]

        return []

    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position for a symbol."""
        result = await self._request("GET", f"/v2/positions/{symbol.upper()}")

        if result:
            return {
                "symbol": result.get("symbol"),
                "quantity": int(result.get("qty", 0)),
                "avg_entry_price": Decimal(str(result.get("avg_entry_price", 0))),
                "market_value": Decimal(str(result.get("market_value", 0))),
                "unrealized_pl": Decimal(str(result.get("unrealized_pl", 0))),
                "current_price": Decimal(str(result.get("current_price", 0))),
            }

        return None

    async def close_position(
        self,
        symbol: str,
        quantity: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Close a position."""
        if quantity:
            # Partial close - place opposite order
            position = await self.get_position(symbol)
            if position:
                side = "sell" if position["quantity"] > 0 else "buy"
                return await self.place_order(
                    symbol=symbol,
                    side=side.upper(),
                    quantity=quantity,
                    order_type="MARKET",
                )
        else:
            # Full close
            result = await self._request("DELETE", f"/v2/positions/{symbol.upper()}")
            if result:
                return {"status": "success", "symbol": symbol, "order": result}

        return {"status": "error", "error": "Close position failed"}

    async def close_all_positions(self) -> List[Dict[str, Any]]:
        """Close all positions."""
        result = await self._request("DELETE", "/v2/positions")

        if isinstance(result, list):
            return result

        return []

    # ─────────────────────────────────────────────────────────────────────────
    # MARKET DATA
    # ─────────────────────────────────────────────────────────────────────────

    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get current quote for a symbol."""
        # Use latest trade and quote endpoints
        result = await self._request(
            "GET",
            f"/v2/stocks/{symbol.upper()}/quotes/latest",
            base_url=self.data_url,
        )

        if result and "quote" in result:
            q = result["quote"]
            return {
                "symbol": symbol.upper(),
                "bid": Decimal(str(q.get("bp", 0))),
                "bid_size": q.get("bs", 0),
                "ask": Decimal(str(q.get("ap", 0))),
                "ask_size": q.get("as", 0),
                "timestamp": q.get("t"),
            }

        return {}

    async def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get quotes for multiple symbols."""
        result = {}
        for symbol in symbols:
            quote = await self.get_quote(symbol)
            if quote:
                result[symbol] = quote
        return result

    async def get_last_trade(self, symbol: str) -> Dict[str, Any]:
        """Get last trade for a symbol."""
        result = await self._request(
            "GET",
            f"/v2/stocks/{symbol.upper()}/trades/latest",
            base_url=self.data_url,
        )

        if result and "trade" in result:
            t = result["trade"]
            return {
                "symbol": symbol.upper(),
                "price": Decimal(str(t.get("p", 0))),
                "size": t.get("s", 0),
                "timestamp": t.get("t"),
            }

        return {}

    async def get_historical_data(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Get historical OHLCV data."""
        # Map interval to Alpaca timeframe
        timeframe_map = {
            "1min": "1Min",
            "5min": "5Min",
            "15min": "15Min",
            "30min": "30Min",
            "1hour": "1Hour",
            "1day": "1Day",
            "day": "1Day",
        }
        timeframe = timeframe_map.get(interval.lower(), "1Day")

        params = {
            "start": start.isoformat() + "Z",
            "end": end.isoformat() + "Z",
            "timeframe": timeframe,
            "limit": 10000,
            "adjustment": "split",  # Adjust for splits
        }

        result = await self._request(
            "GET",
            f"/v2/stocks/{symbol.upper()}/bars",
            params=params,
            base_url=self.data_url,
        )

        if result and "bars" in result:
            bars = result["bars"]
            if not bars:
                return pd.DataFrame()

            df = pd.DataFrame(bars)
            df["timestamp"] = pd.to_datetime(df["t"])
            df.set_index("timestamp", inplace=True)
            df = df.rename(columns={
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            })
            df = df[["open", "high", "low", "close", "volume"]]

            return df

        return pd.DataFrame()

    # ─────────────────────────────────────────────────────────────────────────
    # SYMBOL INFORMATION
    # ─────────────────────────────────────────────────────────────────────────

    async def get_tradable_symbols(self) -> List[str]:
        """Get list of tradable symbols."""
        result = await self._request(
            "GET",
            "/v2/assets",
            params={"status": "active", "asset_class": "us_equity"},
        )

        if isinstance(result, list):
            return [
                a["symbol"]
                for a in result
                if a.get("tradable", False)
            ]

        return []

    async def is_tradable(self, symbol: str) -> bool:
        """Check if a symbol is tradable."""
        result = await self._request("GET", f"/v2/assets/{symbol.upper()}")

        if result:
            return (
                result.get("tradable", False) and
                result.get("status") == "active"
            )

        return False

    async def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get symbol information."""
        result = await self._request("GET", f"/v2/assets/{symbol.upper()}")

        if result:
            return {
                "symbol": result.get("symbol"),
                "name": result.get("name"),
                "exchange": result.get("exchange"),
                "asset_class": result.get("class"),
                "tradable": result.get("tradable"),
                "marginable": result.get("marginable"),
                "shortable": result.get("shortable"),
                "fractionable": result.get("fractionable"),
                "min_order_size": 1,
                "min_trade_increment": 1,
            }

        return {}


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def create_alpaca_broker(
    api_key: str,
    api_secret: str,
    paper: bool = True,
) -> AlpacaBroker:
    """Create an Alpaca broker instance."""
    return AlpacaBroker(api_key, api_secret, paper)
