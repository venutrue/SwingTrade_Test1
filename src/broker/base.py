"""
Broker Interface and Connection Management
==========================================
Abstract broker interface with connection resilience and reconnection logic.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List, Callable
from enum import Enum
import logging

import pandas as pd

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION STATE
# ─────────────────────────────────────────────────────────────────────────────

class ConnectionState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    ERROR = "ERROR"


# ─────────────────────────────────────────────────────────────────────────────
# BROKER INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class BaseBroker(ABC):
    """
    Abstract base class for broker implementations.

    Provides:
    - Connection lifecycle management
    - Automatic reconnection with exponential backoff
    - Rate limiting
    - Health monitoring
    """

    def __init__(self):
        self.settings = get_settings()
        self.state = ConnectionState.DISCONNECTED
        self.connected_at: Optional[datetime] = None
        self.last_heartbeat: Optional[datetime] = None
        self.reconnect_attempts = 0

        self._connection_callbacks: List[Callable] = []
        self._disconnection_callbacks: List[Callable] = []
        self._reconnection_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._rate_limiter = RateLimiter(
            max_requests=self.settings.broker.max_requests_per_second,
            time_window=1.0
        )

    # ─────────────────────────────────────────────────────────────────────────
    # CONNECTION MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    async def _connect_impl(self) -> bool:
        """Implementation-specific connection logic."""
        pass

    @abstractmethod
    async def _disconnect_impl(self) -> None:
        """Implementation-specific disconnection logic."""
        pass

    @abstractmethod
    async def _health_check_impl(self) -> bool:
        """Implementation-specific health check."""
        pass

    async def connect(self) -> bool:
        """
        Connect to the broker with retry logic.
        """
        if self.state == ConnectionState.CONNECTED:
            return True

        self.state = ConnectionState.CONNECTING
        max_retries = self.settings.broker.max_retries
        retry_delay = self.settings.broker.retry_delay_seconds

        for attempt in range(max_retries):
            try:
                logger.info(f"Connecting to broker (attempt {attempt + 1}/{max_retries})")

                if await self._connect_impl():
                    self.state = ConnectionState.CONNECTED
                    self.connected_at = datetime.now()
                    self.reconnect_attempts = 0

                    # Start heartbeat monitoring
                    await self._start_heartbeat()

                    # Notify callbacks
                    await self._notify_connected()

                    logger.info("Successfully connected to broker")
                    return True

            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)

        self.state = ConnectionState.ERROR
        logger.error("Failed to connect to broker after all retries")
        return False

    async def disconnect(self) -> None:
        """Disconnect from broker."""
        if self.state == ConnectionState.DISCONNECTED:
            return

        logger.info("Disconnecting from broker")

        # Stop heartbeat
        await self._stop_heartbeat()

        # Stop reconnection task if running
        if self._reconnection_task:
            self._reconnection_task.cancel()
            try:
                await self._reconnection_task
            except asyncio.CancelledError:
                pass

        try:
            await self._disconnect_impl()
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

        self.state = ConnectionState.DISCONNECTED
        self.connected_at = None

        # Notify callbacks
        await self._notify_disconnected()

    async def reconnect(self) -> bool:
        """
        Reconnect to broker after connection loss.
        """
        if self.state == ConnectionState.RECONNECTING:
            return False

        self.state = ConnectionState.RECONNECTING
        self.reconnect_attempts += 1

        logger.warning(f"Attempting reconnection (attempt {self.reconnect_attempts})")

        try:
            await self._disconnect_impl()
        except Exception:
            pass

        return await self.connect()

    async def _start_heartbeat(self) -> None:
        """Start heartbeat monitoring task."""
        if self._heartbeat_task:
            return

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _stop_heartbeat(self) -> None:
        """Stop heartbeat monitoring task."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

    async def _heartbeat_loop(self) -> None:
        """Heartbeat loop to monitor connection health."""
        interval = self.settings.monitoring.health_check_interval_seconds

        while True:
            try:
                await asyncio.sleep(interval)

                if self.state != ConnectionState.CONNECTED:
                    continue

                # Perform health check
                if await self._health_check_impl():
                    self.last_heartbeat = datetime.now()
                else:
                    logger.warning("Health check failed, initiating reconnection")
                    asyncio.create_task(self.reconnect())

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # CALLBACKS
    # ─────────────────────────────────────────────────────────────────────────

    def on_connected(self, callback: Callable) -> None:
        """Register callback for connection events."""
        self._connection_callbacks.append(callback)

    def on_disconnected(self, callback: Callable) -> None:
        """Register callback for disconnection events."""
        self._disconnection_callbacks.append(callback)

    async def _notify_connected(self) -> None:
        """Notify all connection callbacks."""
        for callback in self._connection_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Error in connection callback: {e}")

    async def _notify_disconnected(self) -> None:
        """Notify all disconnection callbacks."""
        for callback in self._disconnection_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Error in disconnection callback: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # RATE LIMITING
    # ─────────────────────────────────────────────────────────────────────────

    async def _rate_limit(self) -> None:
        """Apply rate limiting before API calls."""
        await self._rate_limiter.acquire()

    # ─────────────────────────────────────────────────────────────────────────
    # ABSTRACT METHODS (To be implemented by specific brokers)
    # ─────────────────────────────────────────────────────────────────────────

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        product_type: str = "CNC",
    ) -> Dict[str, Any]:
        """Place an order."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order."""
        pass

    @abstractmethod
    async def modify_order(
        self,
        order_id: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Modify an existing order."""
        pass

    @abstractmethod
    async def get_orders(self) -> List[Dict[str, Any]]:
        """Get all orders for the day."""
        pass

    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific order by ID."""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        pass

    @abstractmethod
    async def get_holdings(self) -> List[Dict[str, Any]]:
        """Get portfolio holdings."""
        pass

    @abstractmethod
    async def get_balance(self) -> Dict[str, Any]:
        """Get account balance/margins."""
        pass

    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        interval: str,
        from_date: datetime,
        to_date: datetime,
    ) -> pd.DataFrame:
        """Get historical OHLCV data."""
        pass

    @abstractmethod
    async def get_quote(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get real-time quotes for symbols."""
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # STATUS
    # ─────────────────────────────────────────────────────────────────────────

    def is_connected(self) -> bool:
        """Check if broker is connected."""
        return self.state == ConnectionState.CONNECTED

    def get_status(self) -> Dict[str, Any]:
        """Get broker connection status."""
        return {
            "state": self.state.value,
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "reconnect_attempts": self.reconnect_attempts,
        }


# ─────────────────────────────────────────────────────────────────────────────
# RATE LIMITER
# ─────────────────────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    """

    def __init__(self, max_requests: int, time_window: float = 1.0):
        self.max_requests = max_requests
        self.time_window = time_window
        self.tokens = max_requests
        self.last_refill = datetime.now()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = datetime.now()
            elapsed = (now - self.last_refill).total_seconds()

            # Refill tokens based on elapsed time
            self.tokens = min(
                self.max_requests,
                self.tokens + (elapsed / self.time_window) * self.max_requests
            )
            self.last_refill = now

            if self.tokens < 1:
                # Calculate wait time
                wait_time = (1 - self.tokens) * self.time_window / self.max_requests
                await asyncio.sleep(wait_time)
                self.tokens = 1

            self.tokens -= 1
