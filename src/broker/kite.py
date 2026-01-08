"""
Zerodha Kite Connect Broker Implementation
==========================================
Production-grade Kite Connect integration with WebSocket support.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List, Callable
import logging

import pandas as pd

from src.broker.base import BaseBroker, ConnectionState
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# KITE CONNECT BROKER
# ─────────────────────────────────────────────────────────────────────────────

class KiteConnectBroker(BaseBroker):
    """
    Production Zerodha Kite Connect implementation.

    Features:
    - Async API calls with connection pooling
    - Automatic token refresh
    - WebSocket tick data
    - Rate limiting
    - Error handling and retries
    """

    def __init__(self):
        super().__init__()
        self.kite = None
        self.kws = None  # KiteTicker for WebSocket
        self._instrument_cache: Dict[str, int] = {}
        self._tick_callbacks: List[Callable] = []
        self._order_update_callbacks: List[Callable] = []

    # ─────────────────────────────────────────────────────────────────────────
    # CONNECTION IMPLEMENTATION
    # ─────────────────────────────────────────────────────────────────────────

    async def _connect_impl(self) -> bool:
        """Connect to Kite API."""
        try:
            # Import here to handle missing dependency gracefully
            from kiteconnect import KiteConnect

            settings = self.settings.broker

            self.kite = KiteConnect(api_key=settings.api_key)

            # Set access token if available
            if settings.access_token:
                self.kite.set_access_token(settings.access_token)
                logger.info("Using existing access token")
            elif settings.request_token:
                # Generate session from request token
                data = self.kite.generate_session(
                    settings.request_token,
                    api_secret=settings.api_secret
                )
                settings.access_token = data["access_token"]
                self.kite.set_access_token(settings.access_token)
                logger.info("Generated new access token from request token")
            else:
                # Need user to authenticate
                login_url = self.kite.login_url()
                logger.warning(f"Authentication required. Visit: {login_url}")
                return False

            # Verify connection by fetching profile
            profile = self.kite.profile()
            logger.info(f"Connected as: {profile.get('user_name', 'Unknown')}")

            # Cache instruments
            await self._cache_instruments()

            return True

        except Exception as e:
            logger.error(f"Kite connection failed: {e}")
            return False

    async def _disconnect_impl(self) -> None:
        """Disconnect from Kite API."""
        try:
            if self.kws:
                self.kws.close()
                self.kws = None

            # Invalidate session
            if self.kite:
                try:
                    self.kite.invalidate_access_token()
                except Exception:
                    pass
                self.kite = None

        except Exception as e:
            logger.error(f"Error during Kite disconnect: {e}")

    async def _health_check_impl(self) -> bool:
        """Check Kite API health."""
        try:
            if not self.kite:
                return False

            # Try to fetch margins as health check
            margins = self.kite.margins()
            return margins is not None

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def _cache_instruments(self) -> None:
        """Cache instrument tokens for faster lookups."""
        try:
            instruments = self.kite.instruments("NSE")
            for inst in instruments:
                self._instrument_cache[inst["tradingsymbol"]] = inst["instrument_token"]
            logger.info(f"Cached {len(self._instrument_cache)} instruments")
        except Exception as e:
            logger.error(f"Failed to cache instruments: {e}")

    def _get_instrument_token(self, symbol: str, exchange: str = "NSE") -> int:
        """Get instrument token for a symbol."""
        # Check cache first
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]

        # Fetch from API
        instruments = self.kite.instruments(exchange)
        for inst in instruments:
            if inst["tradingsymbol"] == symbol:
                self._instrument_cache[symbol] = inst["instrument_token"]
                return inst["instrument_token"]

        raise ValueError(f"Symbol {symbol} not found on {exchange}")

    # ─────────────────────────────────────────────────────────────────────────
    # ORDER MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        product_type: str = "CNC",
        variety: str = "regular",
        exchange: str = "NSE",
        validity: str = "DAY",
        disclosed_quantity: int = 0,
        tag: str = "",
    ) -> Dict[str, Any]:
        """
        Place an order on Kite.

        Returns:
            dict with order_id and status
        """
        await self._rate_limit()

        if not self.kite:
            return {"status": "error", "error": "Not connected"}

        try:
            # Map order type
            kite_order_type = {
                "MARKET": self.kite.ORDER_TYPE_MARKET,
                "LIMIT": self.kite.ORDER_TYPE_LIMIT,
                "SL": self.kite.ORDER_TYPE_SL,
                "SL-M": self.kite.ORDER_TYPE_SLM,
            }.get(order_type, self.kite.ORDER_TYPE_MARKET)

            # Map transaction type
            transaction_type = (
                self.kite.TRANSACTION_TYPE_BUY if side == "BUY"
                else self.kite.TRANSACTION_TYPE_SELL
            )

            # Map product type
            kite_product = {
                "CNC": self.kite.PRODUCT_CNC,
                "MIS": self.kite.PRODUCT_MIS,
                "NRML": self.kite.PRODUCT_NRML,
            }.get(product_type, self.kite.PRODUCT_CNC)

            # Map variety
            kite_variety = {
                "regular": self.kite.VARIETY_REGULAR,
                "amo": self.kite.VARIETY_AMO,
                "bo": self.kite.VARIETY_BO,
                "co": self.kite.VARIETY_CO,
                "iceberg": self.kite.VARIETY_ICEBERG,
                "auction": self.kite.VARIETY_AUCTION,
            }.get(variety, self.kite.VARIETY_REGULAR)

            # Build order params
            order_params = {
                "variety": kite_variety,
                "exchange": exchange,
                "tradingsymbol": symbol,
                "transaction_type": transaction_type,
                "quantity": quantity,
                "product": kite_product,
                "order_type": kite_order_type,
                "validity": validity,
            }

            # Add optional params
            if price and order_type in ("LIMIT", "SL"):
                order_params["price"] = price

            if trigger_price and order_type in ("SL", "SL-M"):
                order_params["trigger_price"] = trigger_price

            if disclosed_quantity:
                order_params["disclosed_quantity"] = disclosed_quantity

            if tag:
                order_params["tag"] = tag

            # Place order
            order_id = self.kite.place_order(**order_params)

            logger.info(f"Order placed: {order_id} - {symbol} {side} {quantity}")

            return {
                "status": "success",
                "order_id": str(order_id),
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
            }

        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
            }

    async def cancel_order(
        self,
        order_id: str,
        variety: str = "regular"
    ) -> Dict[str, Any]:
        """Cancel an order."""
        await self._rate_limit()

        if not self.kite:
            return {"status": "error", "error": "Not connected"}

        try:
            kite_variety = {
                "regular": self.kite.VARIETY_REGULAR,
                "amo": self.kite.VARIETY_AMO,
                "bo": self.kite.VARIETY_BO,
                "co": self.kite.VARIETY_CO,
            }.get(variety, self.kite.VARIETY_REGULAR)

            result = self.kite.cancel_order(kite_variety, order_id)

            logger.info(f"Order cancelled: {order_id}")

            return {
                "status": "success",
                "order_id": order_id,
                "result": result,
            }

        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "order_id": order_id,
            }

    async def modify_order(
        self,
        order_id: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        order_type: Optional[str] = None,
        variety: str = "regular",
    ) -> Dict[str, Any]:
        """Modify an existing order."""
        await self._rate_limit()

        if not self.kite:
            return {"status": "error", "error": "Not connected"}

        try:
            kite_variety = {
                "regular": self.kite.VARIETY_REGULAR,
                "amo": self.kite.VARIETY_AMO,
                "bo": self.kite.VARIETY_BO,
                "co": self.kite.VARIETY_CO,
            }.get(variety, self.kite.VARIETY_REGULAR)

            modify_params = {"variety": kite_variety, "order_id": order_id}

            if quantity:
                modify_params["quantity"] = quantity
            if price:
                modify_params["price"] = price
            if trigger_price:
                modify_params["trigger_price"] = trigger_price
            if order_type:
                modify_params["order_type"] = order_type

            result = self.kite.modify_order(**modify_params)

            logger.info(f"Order modified: {order_id}")

            return {
                "status": "success",
                "order_id": order_id,
                "result": result,
            }

        except Exception as e:
            logger.error(f"Order modification failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "order_id": order_id,
            }

    async def get_orders(self) -> List[Dict[str, Any]]:
        """Get all orders for the day."""
        await self._rate_limit()

        if not self.kite:
            return []

        try:
            return self.kite.orders() or []
        except Exception as e:
            logger.error(f"Failed to fetch orders: {e}")
            return []

    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific order by ID."""
        await self._rate_limit()

        if not self.kite:
            return None

        try:
            history = self.kite.order_history(order_id)
            return history[-1] if history else None
        except Exception as e:
            logger.error(f"Failed to fetch order {order_id}: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # POSITIONS AND HOLDINGS
    # ─────────────────────────────────────────────────────────────────────────

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        await self._rate_limit()

        if not self.kite:
            return []

        try:
            positions = self.kite.positions()
            return positions.get("net", []) if positions else []
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    async def get_holdings(self) -> List[Dict[str, Any]]:
        """Get portfolio holdings."""
        await self._rate_limit()

        if not self.kite:
            return []

        try:
            return self.kite.holdings() or []
        except Exception as e:
            logger.error(f"Failed to fetch holdings: {e}")
            return []

    async def get_balance(self) -> Dict[str, Any]:
        """Get account balance/margins."""
        await self._rate_limit()

        if not self.kite:
            return {}

        try:
            margins = self.kite.margins()
            if margins and "equity" in margins:
                equity = margins["equity"]
                return {
                    "available_cash": equity.get("available", {}).get("live_balance", 0),
                    "available_margin": equity.get("available", {}).get("cash", 0),
                    "used_margin": equity.get("utilised", {}).get("debits", 0),
                    "total_margin": equity.get("net", 0),
                }
            return {}
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return {}

    # ─────────────────────────────────────────────────────────────────────────
    # MARKET DATA
    # ─────────────────────────────────────────────────────────────────────────

    async def get_historical_data(
        self,
        symbol: str,
        interval: str,
        from_date: datetime,
        to_date: datetime,
        continuous: bool = False,
        oi: bool = False,
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data.

        Args:
            symbol: Trading symbol
            interval: Candle interval (minute, 3minute, 5minute, 10minute,
                     15minute, 30minute, 60minute, day, week, month)
            from_date: Start date
            to_date: End date
            continuous: For F&O continuous data
            oi: Include open interest

        Returns:
            DataFrame with columns: [open, high, low, close, volume]
        """
        await self._rate_limit()

        if not self.kite:
            return pd.DataFrame()

        try:
            instrument_token = self._get_instrument_token(symbol)

            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
                continuous=continuous,
                oi=oi,
            )

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            df.set_index("date", inplace=True)
            df.columns = ["open", "high", "low", "close", "volume"]
            df.index = pd.to_datetime(df.index)

            return df

        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return pd.DataFrame()

    async def get_quote(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get real-time quotes for symbols."""
        await self._rate_limit()

        if not self.kite:
            return {}

        try:
            # Format symbols for Kite API (EXCHANGE:SYMBOL)
            formatted = [f"NSE:{s}" for s in symbols]
            return self.kite.quote(formatted) or {}
        except Exception as e:
            logger.error(f"Failed to fetch quotes: {e}")
            return {}

    async def get_ltp(self, symbols: List[str]) -> Dict[str, float]:
        """Get last traded price for symbols."""
        await self._rate_limit()

        if not self.kite:
            return {}

        try:
            formatted = [f"NSE:{s}" for s in symbols]
            ltp_data = self.kite.ltp(formatted) or {}

            return {
                symbol: data.get("last_price", 0)
                for symbol, data in ltp_data.items()
            }
        except Exception as e:
            logger.error(f"Failed to fetch LTP: {e}")
            return {}

    # ─────────────────────────────────────────────────────────────────────────
    # WEBSOCKET (TICK DATA)
    # ─────────────────────────────────────────────────────────────────────────

    async def start_ticker(self, symbols: List[str]) -> None:
        """Start WebSocket ticker for real-time data."""
        if not self.kite:
            logger.error("Cannot start ticker: Not connected")
            return

        try:
            from kiteconnect import KiteTicker

            self.kws = KiteTicker(
                self.settings.broker.api_key,
                self.settings.broker.access_token
            )

            # Get instrument tokens
            tokens = [self._get_instrument_token(s) for s in symbols]

            def on_ticks(ws, ticks):
                for tick in ticks:
                    for callback in self._tick_callbacks:
                        try:
                            callback(tick)
                        except Exception as e:
                            logger.error(f"Tick callback error: {e}")

            def on_connect(ws, response):
                logger.info("WebSocket connected")
                ws.subscribe(tokens)
                ws.set_mode(ws.MODE_FULL, tokens)

            def on_close(ws, code, reason):
                logger.warning(f"WebSocket closed: {code} - {reason}")

            def on_error(ws, code, reason):
                logger.error(f"WebSocket error: {code} - {reason}")

            def on_order_update(ws, data):
                for callback in self._order_update_callbacks:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error(f"Order update callback error: {e}")

            self.kws.on_ticks = on_ticks
            self.kws.on_connect = on_connect
            self.kws.on_close = on_close
            self.kws.on_error = on_error
            self.kws.on_order_update = on_order_update

            # Run in background thread (KiteTicker uses threading)
            self.kws.connect(threaded=True)
            logger.info(f"Ticker started for {len(symbols)} symbols")

        except Exception as e:
            logger.error(f"Failed to start ticker: {e}")

    async def stop_ticker(self) -> None:
        """Stop WebSocket ticker."""
        if self.kws:
            self.kws.close()
            self.kws = None
            logger.info("Ticker stopped")

    def on_tick(self, callback: Callable) -> None:
        """Register callback for tick data."""
        self._tick_callbacks.append(callback)

    def on_order_update(self, callback: Callable) -> None:
        """Register callback for order updates."""
        self._order_update_callbacks.append(callback)


# ─────────────────────────────────────────────────────────────────────────────
# PAPER TRADING BROKER
# ─────────────────────────────────────────────────────────────────────────────

class PaperTradingBroker(BaseBroker):
    """
    Paper trading broker for testing without real money.

    Simulates order execution with configurable slippage and delays.
    """

    def __init__(self, initial_capital: float = 1000000.0):
        super().__init__()
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.positions: Dict[str, Dict] = {}
        self.orders: List[Dict] = []
        self.order_counter = 0
        self.price_cache: Dict[str, float] = {}

    async def _connect_impl(self) -> bool:
        """Paper broker is always connected."""
        logger.info(f"Paper trading broker initialized with ₹{self.capital:,.2f}")
        return True

    async def _disconnect_impl(self) -> None:
        """No cleanup needed for paper broker."""
        pass

    async def _health_check_impl(self) -> bool:
        """Paper broker is always healthy."""
        return True

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        product_type: str = "CNC",
        **kwargs
    ) -> Dict[str, Any]:
        """Simulate order placement."""
        self.order_counter += 1
        order_id = f"PAPER_{self.order_counter:06d}"

        # Simulate slippage
        slippage = float(self.settings.risk.expected_slippage_percent) / 100
        execution_price = price or self.price_cache.get(symbol, 100.0)

        if side == "BUY":
            execution_price *= (1 + slippage)
        else:
            execution_price *= (1 - slippage)

        # Calculate order value
        order_value = execution_price * quantity

        # Check capital
        if side == "BUY" and order_value > self.capital:
            return {
                "status": "error",
                "error": "Insufficient capital",
                "order_id": None,
            }

        # Update capital
        if side == "BUY":
            self.capital -= order_value
        else:
            self.capital += order_value

        # Track order
        order = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": execution_price,
            "status": "COMPLETE",
            "filled_quantity": quantity,
            "average_price": execution_price,
            "timestamp": datetime.now().isoformat(),
        }
        self.orders.append(order)

        # Update positions
        if symbol not in self.positions:
            self.positions[symbol] = {"quantity": 0, "average_price": 0}

        pos = self.positions[symbol]
        if side == "BUY":
            total_value = pos["quantity"] * pos["average_price"] + order_value
            pos["quantity"] += quantity
            pos["average_price"] = total_value / pos["quantity"] if pos["quantity"] > 0 else 0
        else:
            pos["quantity"] -= quantity

        logger.info(f"[PAPER] Order filled: {order_id} - {symbol} {side} {quantity}@{execution_price:.2f}")

        return {
            "status": "success",
            "order_id": order_id,
            "execution_price": execution_price,
        }

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a paper order."""
        return {"status": "success", "order_id": order_id}

    async def modify_order(self, order_id: str, **kwargs) -> Dict[str, Any]:
        """Modify a paper order."""
        return {"status": "success", "order_id": order_id}

    async def get_orders(self) -> List[Dict[str, Any]]:
        """Get paper orders."""
        return self.orders

    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific paper order."""
        for order in self.orders:
            if order["order_id"] == order_id:
                return order
        return None

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get paper positions."""
        return [
            {"tradingsymbol": symbol, **pos}
            for symbol, pos in self.positions.items()
            if pos["quantity"] != 0
        ]

    async def get_holdings(self) -> List[Dict[str, Any]]:
        """Get paper holdings (same as positions for paper trading)."""
        return await self.get_positions()

    async def get_balance(self) -> Dict[str, Any]:
        """Get paper balance."""
        return {
            "available_cash": self.capital,
            "available_margin": self.capital,
            "used_margin": self.initial_capital - self.capital,
            "total_margin": self.initial_capital,
        }

    async def get_historical_data(
        self,
        symbol: str,
        interval: str,
        from_date: datetime,
        to_date: datetime,
    ) -> pd.DataFrame:
        """Get historical data (would need real data source)."""
        # For paper trading, return empty - should use real data service
        return pd.DataFrame()

    async def get_quote(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get cached quotes."""
        return {
            f"NSE:{s}": {"last_price": self.price_cache.get(s, 100.0)}
            for s in symbols
        }

    def set_price(self, symbol: str, price: float) -> None:
        """Set price for paper trading simulation."""
        self.price_cache[symbol] = price
